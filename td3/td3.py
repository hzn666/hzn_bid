from typing import Dict, Tuple, List
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from segment_tree import MinSegmentTree, SumSegmentTree


class ReplayBuffer:
    """A simple numpy replay buffer."""

    def __init__(self, obs_dim: int, size: int, batch_size: int = 32):
        self.obs_buf = np.zeros([size, obs_dim], dtype=np.float32)
        self.next_obs_buf = np.zeros([size, obs_dim], dtype=np.float32)
        self.acts_buf = np.zeros([size], dtype=np.float32)
        self.rews_buf = np.zeros([size], dtype=np.float32)
        self.done_buf = np.zeros([size], dtype=np.float32)
        self.max_size = size
        self.batch_size = batch_size
        self.ptr = 0
        self.size = 0

    def store(
            self,
            obs: np.ndarray,
            act: np.ndarray,
            rew: float,
            next_obs: np.ndarray,
            done: bool
    ):
        self.obs_buf[self.ptr] = obs
        self.acts_buf[self.ptr] = act
        self.rews_buf[self.ptr] = rew
        self.next_obs_buf[self.ptr] = next_obs
        self.done_buf[self.ptr] = done

        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.max_size, self.size + 1)

    def sample_batch(self) -> Dict[str, np.ndarray]:
        idxs = np.random.choice(self.size, size=self.batch_size, replace=False)
        return dict(
            obs=self.obs_buf[idxs],
            next_obs=self.next_obs_buf[idxs],
            acts=self.acts_buf[idxs],
            rews=self.rews_buf[idxs],
            done=self.done_buf[idxs]
        )

    def __len__(self) -> int:
        return self.size


class PrioritizedReplayBuffer(ReplayBuffer):

    def __init__(
            self,
            obs_dim: int,
            size: int,
            batch_size: int = 32,
            alpha: float = 0.6
    ):
        """Initialization."""
        assert alpha >= 0

        super(PrioritizedReplayBuffer, self).__init__(obs_dim, size, batch_size)
        self.max_priority = 1.0
        self.tree_ptr = 0
        self.alpha = alpha

        # capacity must be positive and a power of 2
        tree_capacity = 1
        while tree_capacity < self.max_size:
            tree_capacity *= 2

        self.sum_tree = SumSegmentTree(tree_capacity)
        self.min_tree = MinSegmentTree(tree_capacity)

    def store(
            self,
            obs: np.ndarray,
            act: int,
            rew: float,
            next_obs: np.ndarray,
            done: bool
    ):
        """Store experience and priority."""
        super().store(obs, act, rew, next_obs, done)

        self.sum_tree[self.tree_ptr] = self.max_priority ** self.alpha
        self.min_tree[self.tree_ptr] = self.max_priority ** self.alpha
        self.tree_ptr = (self.tree_ptr + 1) % self.max_size

    def sample_batch(self, beta: float = 0.4) -> Dict[str, np.ndarray]:
        """Sample a batch of experience."""
        assert len(self) >= self.batch_size
        assert beta > 0

        indices = self._sample_proportional()

        obs = self.obs_buf[indices]
        next_obs = self.next_obs_buf[indices]
        acts = self.acts_buf[indices]
        rews = self.rews_buf[indices]
        done = self.done_buf[indices]
        weights = np.array([self._calculate_weight(i, beta) for i in indices])

        return dict(
            obs=obs,
            next_obs=next_obs,
            acts=acts,
            rews=rews,
            done=done,
            weights=weights,
            indices=indices,
        )

    def update_priorities(self, indices: List[int], priorities: np.ndarray):
        """Update priorities of sampled transitions."""
        assert len(indices) == len(priorities)

        for idx, priority in zip(indices, priorities):
            # assert priority > 0
            assert 0 <= idx < len(self)

            self.sum_tree[idx] = priority ** self.alpha
            self.min_tree[idx] = priority ** self.alpha

            self.max_priority = max(self.max_priority, priority)

    def _sample_proportional(self) -> List[int]:
        """Sample indices based on proportions."""
        indices = []
        p_total = self.sum_tree.sum(0, len(self) - 1)
        segment = p_total / self.batch_size

        for i in range(self.batch_size):
            a = segment * i
            b = segment * (i + 1)
            upperbound = random.uniform(a, b)
            idx = self.sum_tree.retrieve(upperbound)
            indices.append(idx)

        return indices

    def _calculate_weight(self, idx: int, beta: float):
        """Calculate the weight of the experience at idx."""
        # get max weight
        p_min = self.min_tree.min() / self.sum_tree.sum()
        max_weight = (p_min * len(self)) ** (-beta)

        # calculate weights
        p_sample = self.sum_tree[idx] / self.sum_tree.sum()
        weight = (p_sample * len(self)) ** (-beta)
        weight = weight / max_weight

        return weight


class GaussianNoise:
    """Gaussian Noise."""

    def __init__(
            self,
            action_dim: int,
            min_sigma: float = 1.0,
            max_sigma: float = 1.0,
            decay_period: int = 1000000
    ):
        """Initialize."""
        self.action_dim = action_dim
        self.max_sigma = max_sigma
        self.min_sigma = min_sigma
        self.decay_period = decay_period

    def sample(self, t: int = 0) -> float:
        """Get an action with gaussian noise."""
        sigma = self.max_sigma - (self.max_sigma - self.min_sigma) * min(
            1.0, t / self.decay_period
        )

        return np.random.normal(0, sigma, size=self.action_dim)


class Actor(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, init_w: float = 5e-3):
        """Initialize."""
        super(Actor, self).__init__()

        self.bn_input = nn.BatchNorm1d(4)
        self.bn_input.weight.data.fill_(1)
        self.bn_input.bias.data.zero_()

        self.hidden1 = nn.Linear(in_dim, 128)
        self.hidden2 = nn.Linear(128, 64)
        self.out = nn.Linear(64, out_dim)

        self.out.weight.data.uniform_(-init_w, init_w)
        self.out.bias.data.fill_(0)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Forward method implementation."""
        state = self.bn_input(state)
        x = F.relu(self.hidden1(state))
        x = F.relu(self.hidden2(x))
        action = self.out(x).tanh()

        return action


class Critic(nn.Module):
    def __init__(self, in_dim: int, init_w: float = 3e-3):
        """Initialize."""
        super(Critic, self).__init__()

        self.bn_input = nn.BatchNorm1d(4)
        self.bn_input.weight.data.fill_(1)
        self.bn_input.bias.data.zero_()

        self.hidden1 = nn.Linear(in_dim, 128)
        self.hidden2 = nn.Linear(128, 64)
        self.out = nn.Linear(64, 1)

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """Forward method implementation."""
        state = self.bn_input(state)
        x = torch.cat((state, action), dim=-1)
        x = F.relu(self.hidden1(x))
        x = F.relu(self.hidden2(x))
        value = self.out(x)

        return value


class TD3:

    def __init__(
            self,
            obs_dim,
            action_dim,
            memory_size: int,
            batch_size: int,
            gamma: float = 0.99,
            tau: float = 5e-3,
            beta: float = 0.4,
            beta_max: float = 1.0,
            beta_step: float = 1e-5,
            prior_eps: float = 1e-3,
            exploration_noise: float = 0.1,
            target_policy_noise: float = 0.2,
            target_policy_noise_clip: float = 0.5,
            initial_random_steps: int = 200,
            policy_update_freq: int = 3
    ):
        """Initialize."""
        self.memory = PrioritizedReplayBuffer(obs_dim, memory_size, batch_size)
        self.batch_size = batch_size
        self.gamma = gamma
        self.tau = tau
        self.beta = beta
        self.beta_max = beta_max
        self.beta_step = beta_step
        self.prior_eps = prior_eps
        self.initial_random_steps = initial_random_steps
        self.policy_update_freq = policy_update_freq

        # device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(self.device)

        self.writer = SummaryWriter('tensorboard')

        # noise
        self.exploration_noise = GaussianNoise(
            action_dim, exploration_noise, exploration_noise
        )
        self.target_policy_noise = GaussianNoise(
            action_dim, target_policy_noise, target_policy_noise
        )
        self.target_policy_noise_clip = target_policy_noise_clip

        # networks
        self.actor = Actor(obs_dim, action_dim).to(self.device)
        self.actor_target = Actor(obs_dim, action_dim).to(self.device)
        self.actor_target.load_state_dict(self.actor.state_dict())

        self.critic1 = Critic(obs_dim + action_dim).to(self.device)
        self.critic_target1 = Critic(obs_dim + action_dim).to(self.device)
        self.critic_target1.load_state_dict(self.critic1.state_dict())

        self.critic2 = Critic(obs_dim + action_dim).to(self.device)
        self.critic_target2 = Critic(obs_dim + action_dim).to(self.device)
        self.critic_target2.load_state_dict(self.critic2.state_dict())

        # concat critic parameters to use one optim
        critic_parameters = list(self.critic1.parameters()) + list(self.critic2.parameters())

        # optimizer
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=3e-4)
        self.critic_optimizer = optim.Adam(critic_parameters, lr=1e-3)

        # transition to store in memory
        self.transition = list()

        # total steps count
        self.total_step = 0

        # update step for actor
        self.update_step = 0

        # mode
        self.is_test = False

    def select_action(self, state: np.ndarray) -> np.ndarray:
        """Select an action from the input state."""
        # if initial random action should be conducted
        self.actor.eval()
        if self.total_step < self.initial_random_steps and not self.is_test:
            selected_action = np.random.uniform(-0.99, 0.99)
        else:
            selected_action = (
                self.actor(torch.unsqueeze(torch.as_tensor(state, dtype=torch.float32).to(self.device), dim=0))[0]
                    .detach().cpu().numpy()
            )

        # add noise for exploration during training
        if not self.is_test:
            noise = self.exploration_noise.sample()
            selected_action = np.clip(
                selected_action + noise, -0.99, 0.99
            )

            self.transition = [state, selected_action]
        self.total_step += 1
        self.actor.train()
        return selected_action

    def store(self, reward, next_state, done):
        self.transition += [reward, next_state, done]
        self.memory.store(*self.transition)

    def update_model(self) -> Tuple:
        """Update the model by gradient descent."""
        device = self.device

        sample = self.memory.sample_batch(self.beta)
        weights = torch.as_tensor(sample['weights'].reshape(-1, 1), dtype=torch.float32).to(self.device)
        indices = sample['indices']

        states = torch.as_tensor(sample["obs"], dtype=torch.float32).to(device)
        next_states = torch.as_tensor(sample["next_obs"], dtype=torch.float32).to(device)
        actions = torch.as_tensor(sample["acts"].reshape(-1, 1), dtype=torch.float32).to(device)
        rewards = torch.as_tensor(sample["rews"].reshape(-1, 1), dtype=torch.float32).to(device)
        dones = torch.as_tensor(sample["done"].reshape(-1, 1), dtype=torch.float32).to(device)
        masks = 1 - dones

        self.beta = min(self.beta_max, self.beta + self.beta_step)

        # get actions with noise
        noise = torch.as_tensor(self.target_policy_noise.sample(), dtype=torch.float32).to(device)
        clipped_noise = torch.clamp(noise, -self.target_policy_noise_clip, self.target_policy_noise_clip)

        next_actions = (self.actor_target(next_states) + clipped_noise).clamp(-0.99, 0.99)

        next_values1 = self.critic_target1(next_states, next_actions)
        next_values2 = self.critic_target2(next_states, next_actions)
        next_values = torch.min(next_values1, next_values2)

        curr_returns = rewards + self.gamma * next_values * masks
        curr_returns = curr_returns.detach()

        values1 = self.critic1(states, actions)
        values2 = self.critic2(states, actions)
        critic1_loss = F.mse_loss(values1, curr_returns, reduction="none")
        critic2_loss = F.mse_loss(values2, curr_returns, reduction="none")

        critic_loss = critic1_loss + critic2_loss
        critic_loss = (critic_loss * weights).mean()
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        nn.utils.clip_grad_norm_(self.critic1.parameters(), max_norm=40, norm_type=2)
        nn.utils.clip_grad_norm_(self.critic2.parameters(), max_norm=40, norm_type=2)
        self.critic_optimizer.step()

        loss_for_prior = torch.abs(next_values * 2 - values1 - values2).detach().cpu().numpy() / 2
        new_priorites = loss_for_prior + self.prior_eps
        self.memory.update_priorities(indices, new_priorites)

        if self.total_step % self.policy_update_freq == 0:
            # train actor
            actor_loss = -self.critic1(states, self.actor(states)).mean()

            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=40, norm_type=2)
            self.actor_optimizer.step()

            # target_update
            self._target_soft_update()
        else:
            actor_loss = torch.zeros(1)

        return actor_loss.item(), critic_loss.item()

    def _target_soft_update(self):
        """Soft-update: target = tau*local + (1-tau)*target."""
        tau = self.tau
        for t_param, l_param in zip(
                self.actor_target.parameters(), self.actor.parameters()
        ):
            t_param.data.copy_(tau * l_param.data + (1.0 - tau) * t_param.data)

        for t_param, l_param in zip(
                self.critic_target1.parameters(), self.critic1.parameters()
        ):
            t_param.data.copy_(tau * l_param.data + (1.0 - tau) * t_param.data)

        for t_param, l_param in zip(
                self.critic_target2.parameters(), self.critic2.parameters()
        ):
            t_param.data.copy_(tau * l_param.data + (1.0 - tau) * t_param.data)
