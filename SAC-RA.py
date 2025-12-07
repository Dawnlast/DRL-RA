import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import gymnasium as gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal
import numpy as np
import collections, random
import json


from dataclasses import dataclass
from typing import List, Tuple, Optional

import matplotlib.pyplot as plt


@dataclass
class NetworkParams:
    """Network parameters for the ring attractor.
    All parameter values are taken from:
    Sun, X., Mangan, M., & Yue, S. (2018). An Analysis of a Ring Attractor Model
    for Cue Integration. In Biomimetic and Biohybrid Systems (Living Machines 2018),
    pp 459-470."""
    wEEk: float = 45.0  # Excitatory-to-excitatory connection strength constant
    wIEk: float = 60.0  # Inhibitory-to-excitatory connection strength constant
    wEIk: float = -6.0  # Excitatory-to-inhibitory connection strength constant
    wIIk: float = -1.0  # Inhibitory-to-inhibitory connection strength constant

    exc_threshold: float = -1.5  # Base activation threshold for excitatory neurons
    inh_threshold: float = -7.5  # Base activation threshold for inhibitory neurons

    exc_decay: float = 0.005  # Time decay constant for excitatory neuron activity
    inh_decay: float = 0.00025  # Time decay constant for inhibitory neuron activity

    sigma: float = 120.0  # Connection width parameter


@dataclass
class SimulationParams:
    """Simulation parameters."""
    T: float = 0.05  # Total simulation time
    Ti: float = 0.001  # Initial stabilisation time
    tau: float = 1e-4  # Integration time step
    n_neurons: int = 20  # Number of excitatory neurons


class RingAttractor:
    def __init__(self, sim_params: Optional[SimulationParams] = None,
                 net_params: Optional[NetworkParams] = None):
        # Initialize parameters with defaults if not provided
        self.sim_params = sim_params or SimulationParams()
        self.net_params = net_params or NetworkParams()

        # Calculate number of timesteps for total simulation and initial stabilization
        self.Nt = int(np.floor(self.sim_params.T / self.sim_params.tau))
        self.Nti = int(np.floor(self.sim_params.Ti / self.sim_params.tau))

        # Initialize neuron activation arrays
        self.v = np.zeros((self.sim_params.n_neurons, self.Nt))  # Excitatory neurons
        self.u = np.zeros((1, self.Nt))  # Inhibitory neuron (single central neuron)

        # Set initial activation state for excitatory neurons
        self.v[:, 0] = 0.05 * np.ones(self.sim_params.n_neurons)

        # Calculate preferred orientation angles for each excitatory neuron
        # Evenly spaced around 360 degrees
        self.alpha_n = np.linspace(0, 360 - 360 / self.sim_params.n_neurons,
                                   self.sim_params.n_neurons).reshape(-1, 1)

        # Calculate connection weights:
        # For wEE: Full distance-dependent matrix computed using angular differences
        self.wEE = self._compute_weight_matrix()
        self.wIE = self.net_params.wIEk * np.exp(
            0)  # Inhibitory to excitatory weight, simplified as Inhibitory is placed in the middle of the ring.
        self.wEI = self.net_params.wEIk * np.exp(
            0)  # Excitatory to inhibitory weight, simplified as Inhibitory is placed in the middle of the ring.
        self.wII = self.net_params.wIIk * np.exp(
            0)  # Inhibitory self-connection weight, simplified as Inhibitory is placed in the middle of the ring.

    def _compute_weight_matrix(self) -> np.ndarray:
        # Calculate minimum angular differences between all neuron pairs
        # accounting for circular wrapping at 360 degrees
        diff_matrix = np.minimum(
            np.abs(self.alpha_n - self.alpha_n.T),
            360 - np.abs(self.alpha_n - self.alpha_n.T)
        )
        wEE = np.exp(-diff_matrix ** 2 / (2 * self.net_params.sigma ** 2))

        # Scale weights by kernel strength and normalize by number of neurons
        return wEE * (self.net_params.wEEk / self.sim_params.n_neurons)

    def generate_action_signal(self, Q: float, alpha_a: float, sigma_a: float) -> np.ndarray:
        # Calculate minimum angular difference between each neuron's preferred direction
        # and the action direction, accounting for circular wrapping
        diff = np.min([np.abs(self.alpha_n - alpha_a),
                       360 - np.abs(self.alpha_n - alpha_a)], axis=0)

        # Generate Gaussian signal based on action parameters
        signal = (Q * np.exp(-diff ** 2 / (2 * sigma_a ** 2)) /
                  (np.sqrt(2 * np.pi) * sigma_a))

        # Create time-varying signal matrix
        # Signal is zero during initial stabilization period (0 to Ti)
        # then constant for the remainder of the simulation
        x = np.zeros((self.sim_params.n_neurons, self.Nt))
        x[:, self.Nti:] = np.repeat(signal.reshape(-1, 1),
                                    self.Nt - self.Nti, axis=1)
        return x

    def action_space_integration(self, action_values: List[Tuple[float, float, float]]) -> int:
        # Generate all action signals
        input_signals = [self.generate_action_signal(Q, alpha_a, sigma_a)
                         for Q, alpha_a, sigma_a in action_values]

        # Run integration
        for t in range(1, self.Nt):
            # Sum all action signals
            total_input = sum(signal[:, t - 1] for signal in input_signals)

            # Update excitatory neurons
            network_input = (self.net_params.exc_threshold +
                             np.dot(self.wEE, self.v[:, t - 1]) +
                             self.wEI * self.u[:, t - 1] +
                             total_input)

            self.v[:, t] = self.v[:, t - 1] + (-self.v[:, t - 1] +
                                               np.maximum(0,
                                                          network_input)) * self.sim_params.tau / self.net_params.exc_decay

            # Update inhibitory neuron
            inhibitory_input = (self.net_params.inh_threshold +
                                self.wIE * np.sum(self.v[:, t - 1]) / self.sim_params.n_neurons +
                                self.wII * self.u[:, t - 1])

            self.u[:, t] = self.u[:, t - 1] + (-self.u[:, t - 1] +
                                               np.maximum(0,
                                                          inhibitory_input)) * self.sim_params.tau / self.net_params.inh_decay

        # Convert neural activity to action selection
        max_neuron = np.argmax(self.v[:, -1])
        action_idx = int(max_neuron * len(action_values) / self.sim_params.n_neurons)
        return action_idx


#Hyperparameters
lr_pi           = 0.0005
lr_q            = 0.001
init_alpha      = 0.01
gamma           = 0.98
batch_size      = 32
buffer_limit    = 50000
tau             = 0.01 # for target network soft update
target_entropy  = -1.0 # for automated alpha update
lr_alpha        = 0.001  # for automated alpha update

class ReplayBuffer():
    def __init__(self):
        self.buffer = collections.deque(maxlen=buffer_limit)

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
        
        return torch.tensor(s_lst, dtype=torch.float), torch.tensor(a_lst, dtype=torch.float), \
                torch.tensor(r_lst, dtype=torch.float), torch.tensor(s_prime_lst, dtype=torch.float), \
                torch.tensor(done_mask_lst, dtype=torch.float)
    
    def size(self):
        return len(self.buffer)

class PolicyNet(nn.Module):
    def __init__(self, learning_rate):
        super(PolicyNet, self).__init__()
        self.fc1 = nn.Linear(3, 128)
        self.fc_mu = nn.Linear(128,1)
        self.fc_std  = nn.Linear(128,1)
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)

        self.log_alpha = torch.tensor(np.log(init_alpha))
        self.log_alpha.requires_grad = True
        self.log_alpha_optimizer = optim.Adam([self.log_alpha], lr=lr_alpha)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        mu = self.fc_mu(x)
        std = F.softplus(self.fc_std(x))
        dist = Normal(mu, std)
        action = dist.rsample()
        log_prob = dist.log_prob(action)
        real_action = torch.tanh(action)
        real_log_prob = log_prob - torch.log(1-torch.tanh(action).pow(2) + 1e-7)
        return real_action, real_log_prob

    def train_net(self, q1, q2, mini_batch):
        s, _, _, _, _ = mini_batch
        a, log_prob = self.forward(s)
        entropy = -self.log_alpha.exp() * log_prob

        q1_val, q2_val = q1(s,a), q2(s,a)
        q1_q2 = torch.cat([q1_val, q2_val], dim=1)
        min_q = torch.min(q1_q2, 1, keepdim=True)[0]

        loss = -min_q - entropy # for gradient ascent
        self.optimizer.zero_grad()
        loss.mean().backward()
        self.optimizer.step()

        self.log_alpha_optimizer.zero_grad()
        alpha_loss = -(self.log_alpha.exp() * (log_prob + target_entropy).detach()).mean()
        alpha_loss.backward()
        self.log_alpha_optimizer.step()

class QNet(nn.Module):
    def __init__(self, learning_rate):
        super(QNet, self).__init__()
        self.fc_s = nn.Linear(3, 64)
        self.fc_a = nn.Linear(1,64)
        self.fc_cat = nn.Linear(128,32)
        self.fc_out = nn.Linear(32,1)
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)

    def forward(self, x, a):
        h1 = F.relu(self.fc_s(x))
        h2 = F.relu(self.fc_a(a))
        cat = torch.cat([h1,h2], dim=1)
        q = F.relu(self.fc_cat(cat))
        q = self.fc_out(q)
        return q

    def train_net(self, target, mini_batch):
        s, a, r, s_prime, done = mini_batch
        loss = F.smooth_l1_loss(self.forward(s, a) , target)
        self.optimizer.zero_grad()
        loss.mean().backward()
        self.optimizer.step()

    def soft_update(self, net_target):
        for param_target, param in zip(net_target.parameters(), self.parameters()):
            param_target.data.copy_(param_target.data * (1.0 - tau) + param.data * tau)

def calc_target(pi, q1, q2, mini_batch):
    s, a, r, s_prime, done = mini_batch

    with torch.no_grad():
        a_prime, log_prob= pi(s_prime)
        entropy = -pi.log_alpha.exp() * log_prob
        q1_val, q2_val = q1(s_prime,a_prime), q2(s_prime,a_prime)
        q1_q2 = torch.cat([q1_val, q2_val], dim=1)
        min_q = torch.min(q1_q2, 1, keepdim=True)[0]
        target = r + gamma * done * (min_q + entropy)

    return target


def choose_action_with_RA(pi, state, ring_attractor, K=10):
    """
    使用 PolicyNet 采样 K 个动作，再让 Ring Attractor 选择最终动作。
    """
    candidates = []

    s_tensor = torch.from_numpy(state).float().unsqueeze(0)

    # 采样 K 个动作
    for _ in range(K):
        a, logp = pi(s_tensor)
        a = a.item()
        # 映射到角度 0~360°
        angle = (a + 1.0) * 180.0
        candidates.append((1.0, angle, 20.0))  # (Q, alpha, sigma)

    # 用 RA 选择
    action_idx = ring_attractor.action_space_integration(candidates)

    # 使用 index 获取最终动作
    final_action = candidates[action_idx][1] / 180.0 - 1.0  

    return final_action, candidates


ring_attractor = RingAttractor()

def smooth_curve(values, weight=0.9):
    smoothed = []
    last = values[0]
    for v in values:
        new_val = last * weight + (1 - weight) * v
        smoothed.append(new_val)
        last = new_val
    return smoothed

#单个 seed 的训练
def run_one_seed(seed, num_episodes=800):
    np.random.seed(seed)
    import torch
    torch.manual_seed(seed)

    ring_attractor = RingAttractor()
    env = gym.make('Pendulum-v1')
    memory = ReplayBuffer()

    q1, q2, q1_target, q2_target = QNet(lr_q), QNet(lr_q), QNet(lr_q), QNet(lr_q)
    pi = PolicyNet(lr_pi)

    q1_target.load_state_dict(q1.state_dict())
    q2_target.load_state_dict(q2.state_dict())

    print_interval = 20
    score = 0.0
    score_draw = 0.0
    episode_scores = []

    for n_epi in range(num_episodes):
        s, info = env.reset()
        done = False
        truncated = False
        count = 0

        while count < 200 and not (done or truncated):

            a_final, candidates = choose_action_with_RA(pi, s, ring_attractor, K=10)
            s_prime, r, terminated, truncated, info = env.step([2.0 * a_final])

            done = terminated or truncated
            memory.put((s, a_final, r/10.0, s_prime, terminated))

            score += r
            score_draw += r
            s = s_prime
            count += 1

        if memory.size() > 1000:
            for i in range(20):
                mini_batch = memory.sample(batch_size)
                td_target = calc_target(pi, q1_target, q2_target, mini_batch)
                q1.train_net(td_target, mini_batch)
                q2.train_net(td_target, mini_batch)
                entropy = pi.train_net(q1, q2, mini_batch)
                q1.soft_update(q1_target)
                q2.soft_update(q2_target)

        if n_epi % print_interval == 0 and n_epi != 0:
            print(f"[Seed {seed}] Episode {n_epi}, avg score {score/print_interval:.1f}")
            score = 0.0

        episode_scores.append(score_draw)
        score_draw = 0.0

    env.close()
    return episode_scores



#  多 seed 运行
def run_multi_seed(seeds=list(range(0))):

    all_scores = {}
    min_len = 1e9

    for seed in seeds:
        print("========== Running seed", seed, "==========")
        scores = run_one_seed(seed)
        all_scores[str(seed)] = scores
        min_len = min(min_len, len(scores))  # 对齐长度

    for seed in seeds:
        all_scores[str(seed)] = all_scores[str(seed)][:min_len]


    score_matrix = np.array([all_scores[str(seed)] for seed in seeds])
    avg_curve = np.mean(score_matrix, axis=0)
    np.savez("scores_multi_seed_copy.npz", **all_scores, avg=avg_curve)

    with open("scores_multi_seed_copy.json", "w") as f:
        json.dump({"seeds": all_scores, "avg": avg_curve.tolist()}, f, indent=4)

    print("已保存到 scores_multi_seed_copy.npz 和 scores_multi_seed_copy.json")


    # 绘图
    plt.figure(figsize=(10, 6))

    episodes = np.arange(min_len)

    # 平均曲线
    plt.plot(episodes, avg_curve, label="Mean Score (10 seeds)", linewidth=1)

    smoothed = smooth_curve(avg_curve, weight=0.9)
    plt.plot(episodes, smoothed, label="Smoothed Mean (EMA)", linewidth=3)

    plt.xlabel("Episodes")
    plt.ylabel("Score")
    plt.title("SAC + Ring Attractor (5 seeds average)")
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.show()



run_multi_seed()





















