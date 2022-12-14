import time
from collections import defaultdict
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter


class ReplayBuffer(object):
    """A simple numpy replay buffer."""

    def __init__(self, obs_dim: int, size: int, batch_size: int = 32, reward: bool = False):
        self.reward = reward
        self.obs_buf = np.zeros([size, obs_dim], dtype=np.float32)
        self.acts_buf = np.zeros([size], dtype=np.float32)
        self.rews_buf = np.zeros([size], dtype=np.float32)
        if not self.reward:
            self.next_obs_buf = np.zeros([size, obs_dim], dtype=np.float32)
            self.done_buf = np.zeros([size], dtype=np.float32)
        self.max_size = size
        self.batch_size = batch_size
        self.ptr = 0
        self.size = 0

    def store(self,
              obs: np.ndarray,
              act: np.ndarray,
              rew: float,
              next_obs: np.ndarray,
              done: bool,
              ):
        self.obs_buf[self.ptr] = obs
        self.acts_buf[self.ptr] = act
        self.rews_buf[self.ptr] = rew
        if not self.reward:
            self.next_obs_buf[self.ptr] = next_obs
            self.done_buf[self.ptr] = done
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample_batch(self) -> Dict[str, np.ndarray]:
        idxs = np.random.choice(self.size, size=self.batch_size, replace=False)
        if not self.reward:
            return dict(
                obs=self.obs_buf[idxs],
                next_obs=self.next_obs_buf[idxs],
                acts=self.acts_buf[idxs],
                rews=self.rews_buf[idxs],
                done=self.done_buf[idxs]
            )
        else:
            return dict(
                obs=self.obs_buf[idxs],
                acts=self.acts_buf[idxs],
                rews=self.rews_buf[idxs]
            )

    def __len__(self) -> int:
        return self.size


class Network(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        """Initialization"""
        super(Network, self).__init__()

        self.bn_input = nn.BatchNorm1d(in_dim)
        self.bn_input.weight.data.fill_(1)
        self.bn_input.bias.data.fill_(0)

        self.layers = nn.Sequential(
            nn.Linear(in_dim, 100),
            nn.ReLU(),
            nn.Linear(100, 100),
            nn.ReLU(),
            nn.Linear(100, 100),
            nn.ReLU(),
            nn.Linear(100, out_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward method implementation"""
        return self.layers(self.bn_input(x))


class DQN(object):
    def __init__(
            self,
            budget,
            camp,
            obs_dim,
            action_dim,
            action_space,
            lr,
            memory_size: int,
            batch_size: int,
            target_update: int,
            epsilon_decay: float = 1 / 20000,
            max_epsilon: float = 0.9,
            min_epsilon: float = 0.1,
            gamma: float = 1.,
            seed: int = 1,
            time: str = ''
    ):

        def seed_torch(seed: int):
            torch.manual_seed(seed)
            if torch.backends.cudnn.enabled:
                torch.backends.cudnn.benchmark = False
                torch.backends.cudnn.deterministic = True

        np.random.seed(seed)
        seed_torch(seed)

        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.action_space = action_space
        self.lr = lr
        self.memory = ReplayBuffer(obs_dim, memory_size, batch_size)
        self.batch_size = batch_size
        self.epsilon = max_epsilon
        self.epsilon_decay = epsilon_decay
        self.max_epsilon = max_epsilon
        self.min_epsilon = min_epsilon
        self.epsilon_step = 0
        self.target_update = target_update
        self.gamma = gamma

        self.update = 0

        # device: cpu / gpu
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(self.device)

        # networks: dqn, dqn_target, reward_net
        self.dqn = Network(obs_dim, action_dim).to(self.device)
        self.dqn_target = Network(obs_dim, action_dim).to(self.device)
        self.dqn_target.load_state_dict(self.dqn.state_dict())
        self.dqn_target.eval()
        # optimizer
        # self.optimizer = optim.Adam(self.dqn.parameters(), lr=self.lr, weight_decay=1e-5)
        self.optimizer = optim.RMSprop(self.dqn.parameters(), momentum=0.95, weight_decay=1e-5)

        self.reward_net = Network(obs_dim + 1, 1).to(self.device)
        # self.reward_optimizer = optim.Adam(self.reward_net.parameters(), lr=self.lr, weight_decay=1e-5)
        self.reward_optimizer = optim.RMSprop(self.reward_net.parameters(), momentum=0.95, weight_decay=1e-5)
        self.reward_memory = ReplayBuffer(obs_dim, memory_size, batch_size, reward=True)
        self.state_action_reward = defaultdict()

        # transition to store in memory
        self.transition = list()

        # mode: train / test
        self.is_test = False

        # tensorboard
        self.writer = SummaryWriter('tensorboard-camp={}-budget={}-seed={}-{}/'.format(camp, budget, seed, time))
        self.log_cnt = 0
        self.log_cnt_test = 0

    def select_action(self, state: np.ndarray):
        """Select an action from the input state."""
        # epsilon greedy policy
        torch.cuda.empty_cache()
        self.dqn.eval()
        with torch.no_grad():
            if self.epsilon > np.random.random() and not self.is_test:
                index = np.random.randint(0, self.action_dim)
            else:
                q = self.dqn(
                    torch.unsqueeze(torch.as_tensor(state, dtype=torch.float32).to(self.device), dim=0)
                ).argmax()
                index = int(q.detach().cpu().numpy())

        if not self.is_test:
            self.transition = [state, index]
        self.dqn.train()
        return index

    def get_reward(self, state, action):
        self.reward_net.eval()
        state_action = np.hstack((state, action))
        state_action = torch.as_tensor(state_action, dtype=torch.float32).to(self.device)
        state_action = torch.unsqueeze(state_action, dim=0)
        reward = self.reward_net(state_action).detach().cpu().numpy()
        self.reward_net.train()

        return reward

    def store(self, reward, next_state, done):
        self.transition += [reward, next_state, done]
        self.memory.store(*self.transition)

    def update_model(self) -> float:
        """Update the model by gradient descent."""
        samples = self.memory.sample_batch()

        loss = self._compute_dqn_loss(samples)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.update += 1
        self.epsilon = max(
            self.min_epsilon, self.epsilon - (
                    self.max_epsilon - self.min_epsilon
            ) * self.epsilon_decay
        )
        self.epsilon_step += 1
        self.writer.add_scalar('epsilon', self.epsilon, self.epsilon_step)

        if self.update % self.target_update == 0:
            self._target_hard_update()

        return loss.item()

    def update_reward(self) -> float:
        """Update the reward by gradient descent."""
        samples = self.reward_memory.sample_batch()

        loss = self._compute_reward_loss(samples)

        self.reward_optimizer.zero_grad()
        loss.backward()
        self.reward_optimizer.step()

        return loss.item()

    def _compute_dqn_loss(self, samples: Dict[str, np.ndarray]) -> torch.Tensor:
        """Return dqn loss."""
        device = self.device
        state = torch.as_tensor(samples["obs"], dtype=torch.float32).to(device)
        next_state = torch.as_tensor(samples["next_obs"], dtype=torch.float32).to(device)
        action = torch.as_tensor(samples["acts"].reshape(-1, 1), dtype=torch.int64).to(device)
        reward = torch.as_tensor(samples["rews"].reshape(-1, 1), dtype=torch.float32).to(device)
        done = torch.as_tensor(samples["done"].reshape(-1, 1), dtype=torch.float32).to(device)

        # G_t = r + gamma * v(s_{t+1}) if state != Terminal
        #     = r                      otherwise
        curr_q_value = self.dqn(state).gather(1, action)
        next_q_value = self.dqn_target(next_state).max(dim=1, keepdim=True)[0].detach()
        mask = 1 - done
        target = (reward + self.gamma * next_q_value * mask).to(device)

        # calculate dqn loss
        loss = F.mse_loss(curr_q_value, target)

        return loss

    def _compute_reward_loss(self, samples: Dict[str, np.ndarray]) -> torch.Tensor:
        """Return dqn loss."""
        device = self.device
        state = torch.as_tensor(samples["obs"], dtype=torch.float32).to(device)
        action = torch.as_tensor(samples["acts"].reshape(-1, 1), dtype=torch.float32).to(device)
        reward = torch.as_tensor(samples["rews"].reshape(-1, 1), dtype=torch.float32).to(device)

        # G_t = r + gamma * v(s_{t+1}) if state != Terminal
        #     = r                      otherwise
        state_action = torch.hstack((state, action))
        model_reward = self.reward_net(state_action)

        loss = F.mse_loss(model_reward, reward)

        return loss

    def _target_hard_update(self):
        """Hard update: target <- local."""
        self.dqn_target.load_state_dict(self.dqn.state_dict())
