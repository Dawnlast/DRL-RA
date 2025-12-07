import gymnasium as gym
import random
import collections
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
from dataclasses import dataclass
import os

HYPERPARAMS = {
    "lr_mu": 0.0005,
    "lr_q": 0.001,
    "gamma": 0.99,
    "batch_size": 64,
    "buffer_limit": 50000,
    "tau": 0.005,
    "epoch": 80, 
    "start_train_step": 1000,
    "device": torch.device("cuda" if torch.cuda.is_available() else "cpu")
}

class ReplayBuffer:
    def __init__(self):
        self.buffer = collections.deque(maxlen=HYPERPARAMS["buffer_limit"])

    def put(self, transition):
        self.buffer.append(transition)

    def sample(self, n):
        mini_batch = random.sample(self.buffer, n)
        s_lst, a_lst, r_lst, s_prime_lst, done_mask_lst = [], [], [], [], []

        for transition in mini_batch:
            s, a, r, s_prime, done = transition
            s_lst.append(s)
            a_lst.append([a])
            r_lst.append([r])
            s_prime_lst.append(s_prime)
            done_mask = 0.0 if done else 1.0
            done_mask_lst.append([done_mask])

        return (torch.tensor(np.array(s_lst), dtype=torch.float).to(HYPERPARAMS["device"]),
                torch.tensor(np.array(a_lst), dtype=torch.float).to(HYPERPARAMS["device"]),
                torch.tensor(np.array(r_lst), dtype=torch.float).to(HYPERPARAMS["device"]),
                torch.tensor(np.array(s_prime_lst), dtype=torch.float).to(HYPERPARAMS["device"]),
                torch.tensor(np.array(done_mask_lst), dtype=torch.float).to(HYPERPARAMS["device"]))

    def size(self):
        return len(self.buffer)

class MuNet(nn.Module):
    def __init__(self):
        super(MuNet, self).__init__()
        self.fc1 = nn.Linear(3, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc_mu = nn.Linear(64, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        mu = torch.tanh(self.fc_mu(x)) * 2.0 
        return mu

class QNet(nn.Module):
    def __init__(self):
        super(QNet, self).__init__()
        self.fc_s = nn.Linear(3, 64)
        self.fc_a = nn.Linear(1, 64)
        self.fc_q = nn.Linear(128, 32)
        self.fc_out = nn.Linear(32, 1)

    def forward(self, x, a):
        h1 = F.relu(self.fc_s(x))
        h2 = F.relu(self.fc_a(a))
        cat = torch.cat([h1, h2], dim=1)
        q = F.relu(self.fc_q(cat))
        q = self.fc_out(q)
        return q

def soft_update(net, net_target):
    for param_target, param in zip(net_target.parameters(), net.parameters()):
        param_target.data.copy_(param_target.data * (1.0 - HYPERPARAMS["tau"]) + param.data * HYPERPARAMS["tau"])

def train_step(mu, mu_target, q, q_target, memory, q_optimizer, mu_optimizer):
    s, a, r, s_prime, done_mask = memory.sample(HYPERPARAMS["batch_size"])
    
    with torch.no_grad():
        target = r + HYPERPARAMS["gamma"] * q_target(s_prime, mu_target(s_prime)) * done_mask
    
    q_loss = F.smooth_l1_loss(q(s, a), target)
    q_optimizer.zero_grad()
    q_loss.backward()
    q_optimizer.step()
    
    mu_loss = -q(s, mu(s)).mean()
    mu_optimizer.zero_grad()
    mu_loss.backward()
    mu_optimizer.step()

class OrnsteinUhlenbeckNoise:
    def __init__(self, mu):
        self.theta, self.dt, self.sigma = 0.15, 0.01, 0.2
        self.mu = mu
        self.x_prev = np.zeros_like(self.mu)

    def __call__(self):
        x = self.x_prev + self.theta * (self.mu - self.x_prev) * self.dt + \
            self.sigma * np.sqrt(self.dt) * np.random.normal(size=self.mu.shape)
        self.x_prev = x
        return x

@dataclass
class RingAttractorParams:
    tau: float = 1e-4
    T: float = 0.05
    n_neurons: int = 128
    wEEk: float = 45.0
    wIEk: float = 60.0
    wEIk: float = -6.0
    wIIk: float = -1.0
    sigma: float = 15.0 

class RingAttractorLayer:
    def __init__(self, action_range=(-2.0, 2.0)):
        self.p = RingAttractorParams()
        self.Nt = int(self.p.T / self.p.tau)
        self.action_min, self.action_max = action_range
        self.alpha_n = np.linspace(0, 180, self.p.n_neurons)
        diff = np.abs(self.alpha_n[:, None] - self.alpha_n[None, :])
        self.wEE = np.exp(-diff**2 / (2 * self.p.sigma**2)) * (self.p.wEEk / self.p.n_neurons)
    
    def run(self, anchors_q, anchors_a):
        v = np.zeros(self.p.n_neurons) + 0.05 
        u = 0.0
        
        q_min, q_max = np.min(anchors_q), np.max(anchors_q)
        if q_max - q_min < 1e-5:
            norm_q = np.ones_like(anchors_q) 
        else:
            norm_q = (anchors_q - q_min) / (q_max - q_min)
            
        angles = (anchors_a - self.action_min) / (self.action_max - self.action_min) * 180.0
        
        I_ext = np.zeros(self.p.n_neurons)
        for q_val, ang in zip(norm_q, angles):
            diff = np.abs(self.alpha_n - ang)
            I_ext += (q_val * 40.0 + 5.0) * np.exp(-diff**2 / (2 * 30.0**2))
            
        dt_tau_exc = self.p.tau / 0.005
        dt_tau_inh = self.p.tau / 0.00025
        
        for _ in range(self.Nt):
            exc_in = np.dot(self.wEE, v) + self.p.wEIk * u + I_ext - 1.5
            v += (-v + np.maximum(0, exc_in)) * dt_tau_exc
            inh_in = self.p.wIEk * np.mean(v) + self.p.wIIk * u - 7.5
            u += (-u + np.maximum(0, inh_in)) * dt_tau_inh
        
        theta_rad = np.deg2rad(self.alpha_n)
        pop_vector = np.sum(v * np.exp(1j * theta_rad))
        final_angle_rad = np.angle(pop_vector)
        if final_angle_rad < 0:
            final_angle_rad = 0.0
        final_angle = np.rad2deg(final_angle_rad)
        norm_act = final_angle / 180.0
        return np.clip(norm_act * (self.action_max - self.action_min) + self.action_min, self.action_min, self.action_max)

def run_agent(mode="std", seed=42):
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    
    env = gym.make('Pendulum-v1', max_episode_steps=200)
    
    q, q_target = QNet().to(HYPERPARAMS["device"]), QNet().to(HYPERPARAMS["device"])
    mu, mu_target = MuNet().to(HYPERPARAMS["device"]), MuNet().to(HYPERPARAMS["device"])
    q_target.load_state_dict(q.state_dict())
    mu_target.load_state_dict(mu.state_dict())
    
    mu_optimizer = optim.Adam(mu.parameters(), lr=HYPERPARAMS["lr_mu"])
    q_optimizer  = optim.Adam(q.parameters(), lr=HYPERPARAMS["lr_q"])
    
    memory = ReplayBuffer()
    ou_noise = OrnsteinUhlenbeckNoise(mu=np.zeros(1))
    ra_layer = RingAttractorLayer()
    
    score_history = []
    
    for i in range(HYPERPARAMS["epoch"]):
        s, _ = env.reset(seed=seed+i) 
        done = False
        score = 0.0
        
        while not done:
            if mode == "std":
                a_tensor = mu(torch.from_numpy(s).float().to(HYPERPARAMS["device"]))
                a = a_tensor.item() + ou_noise()[0]
                a = np.clip(a, -2.0, 2.0)

            elif mode == "ra":
                with torch.no_grad():
                    state_t = torch.from_numpy(s).float().to(HYPERPARAMS["device"]).unsqueeze(0)
                    base_a = mu(state_t).item()
                    
                    anchors = [base_a] 
                    anchors += [np.clip(base_a + np.random.normal(0, 0.5), -2, 2) for _ in range(4)]
                    anchors += [np.random.uniform(-2, 2) for _ in range(3)]
                    
                    cand_tensor = torch.tensor(anchors, dtype=torch.float).to(HYPERPARAMS["device"]).unsqueeze(1)
                    s_expanded = state_t.repeat(len(anchors), 1)
                    q_vals = q(s_expanded, cand_tensor).cpu().numpy().flatten()

                a = ra_layer.run(q_vals, np.array(anchors))
            
            s_prime, r, terminated, truncated, _ = env.step([a])
            done = terminated or truncated
            
            memory.put((s, a, r/100.0, s_prime, done)) 
            s = s_prime
            score += r
            
            if memory.size() > HYPERPARAMS["start_train_step"]:
                train_step(mu, mu_target, q, q_target, memory, q_optimizer, mu_optimizer)
                soft_update(mu, mu_target)
                soft_update(q, q_target)
                
        score_history.append(score)
        print(f"{mode}-Epoch {i+1}, Score: {score}")
        
    env.close()
    return score_history

def main():
    seeds = [42]
    print(f"Running experiments on {len(seeds)} seeds: {seeds}")
    
    results_std = []
    results_ra = []
    
    for seed in seeds:
        print(f"Running Seed {seed}...")
        results_std.append(run_agent("std", seed))
        results_ra.append(run_agent("ra", seed))
    
    results_std = np.array(results_std)
    results_ra = np.array(results_ra)
    
    mean_std = np.mean(results_std, axis=0)
    std_std = np.std(results_std, axis=0)
    
    mean_ra = np.mean(results_ra, axis=0)
    std_ra = np.std(results_ra, axis=0)
    
    epochs = np.arange(HYPERPARAMS["epoch"])
    
    def smooth(data, window=5):
        return np.convolve(data, np.ones(window)/window, mode='same')

    smooth_mean_std = smooth(mean_std)
    smooth_mean_ra = smooth(mean_ra)

    plt.figure(figsize=(12, 6))
    plt.plot(epochs, smooth_mean_std, label='Standard DDPG', color='blue')
    plt.fill_between(epochs, 
                     mean_std - std_std, 
                     mean_std + std_std, 
                     color='blue', alpha=0.15)
    
    plt.plot(epochs, smooth_mean_ra, label='RA-DDPG', color='orange')
    plt.fill_between(epochs, 
                     mean_ra - std_ra, 
                     mean_ra + std_ra, 
                     color='orange', alpha=0.15)
    
    plt.title(f"Performance Comparison ({len(seeds)} Seeds Average)")
    plt.xlabel("Episode")
    plt.ylabel("Score")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    main()
