import copy
import random
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

from torch.utils.tensorboard import SummaryWriter


class ReplayBuffer:
    """A simple numpy replay buffer."""

    def __init__(self, obs_dim: int, action_dim: int, size: int, batch_size: int = 32):
        """Initialization."""
        self.obs_buf = np.zeros([size, obs_dim], dtype=np.float32)
        self.next_obs_buf = np.zeros([size, obs_dim], dtype=np.float32)
        self.acts_buf = np.zeros([size, action_dim], dtype=np.float32)
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
            done: bool,
    ):
        """Store the transition in buffer."""
        self.obs_buf[self.ptr] = obs
        self.acts_buf[self.ptr] = act
        self.rews_buf[self.ptr] = rew
        self.next_obs_buf[self.ptr] = next_obs
        self.done_buf[self.ptr] = done

        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample_batch(self) -> Dict[str, np.ndarray]:
        """Randomly sample a batch of experiences from memory."""
        idxs = np.random.choice(self.size, size=self.batch_size, replace=False)
        return dict(obs=self.obs_buf[idxs],
                    acts=self.acts_buf[idxs],
                    rews=self.rews_buf[idxs],
                    next_obs=self.next_obs_buf[idxs],
                    done=self.done_buf[idxs])

    def __len__(self) -> int:
        return self.size


class OUNoise:
    """Ornstein-Uhlenbeck process.
    Taken from Udacity deep-reinforcement-learning github repository:
    https://github.com/udacity/deep-reinforcement-learning/blob/master/
    ddpg-pendulum/ddpg_agent.py
    """

    def __init__(
            self,
            size: int,
            mu: float = 0.0,
            theta: float = 0.15,
            sigma: float = 0.2
    ):
        """Initialize parameters and noise process."""
        self.state = np.float64(0.0)
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        self.reset()

    def reset(self):
        """Reset the internal state (=noise) to mean (mu)."""
        self.state = copy.copy(self.mu)

    def sample(self) -> np.ndarray:
        """Update internal state and return it as a noise sample."""
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.array(
            [random.random() for _ in range(len(x))]
        )
        self.state = x + dx
        return self.state


class Actor(nn.Module):
    def __init__(
            self,
            obs_dim: int,
            action_dim: int,
            action_parameters_dim: int,
    ):
        """Initialization."""
        super(Actor, self).__init__()

        self.layers = nn.Sequential(
            nn.Linear(obs_dim, 128),
            nn.Linear(128, 64)
        )
        for layer in self.layers:
            nn.init.kaiming_normal_(layer.weight.data, nonlinearity="leaky_relu")
            nn.init.zeros_(layer.bias.data)

        self.action_output_layer = nn.Linear(64, action_dim)
        nn.init.normal_(self.action_output_layer.weight.data, std=0.01)
        nn.init.zeros_(self.action_output_layer.bias.data)

        self.action_parameters_output_layer = nn.Linear(64, action_parameters_dim)
        nn.init.normal_(self.action_parameters_output_layer.weight.data, std=0.01)
        nn.init.zeros_(self.action_parameters_output_layer.bias.data)

    def forward(self, state: torch.Tensor):
        """Forward method implementation."""
        x = state
        negative_slope = 0.01
        for layer in self.layers:
            x = F.leaky_relu(layer(x), negative_slope)

        action = self.action_output_layer(x)
        action_params = self.action_parameters_output_layer(x).tanh()

        return action, action_params


class Critic(nn.Module):
    def __init__(
            self,
            obs_dim: int,
            action_dim: int,
            action_parameters_dim: int,
    ):
        """Initialization."""
        super(Critic, self).__init__()

        input_dim = obs_dim + action_dim + action_parameters_dim
        self.layers = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.Linear(128, 64)
        )
        for layer in self.layers:
            nn.init.kaiming_normal_(layer.weight.data, nonlinearity="leaky_relu")
            nn.init.zeros_(layer.bias.data)

        self.output_layer = nn.Linear(64, 1)
        nn.init.normal_(self.output_layer.weight.data, std=0.01)
        nn.init.zeros_(self.output_layer.bias.data)

    def forward(
            self,
            state: torch.Tensor,
            action: torch.Tensor,
            action_parameters: torch.Tensor
    ) -> torch.Tensor:
        """Forward method implementation."""
        x = torch.cat((state, action, action_parameters), dim=-1)
        negative_slope = 0.01
        for layer in self.layers:
            x = F.leaky_relu(layer(x), negative_slope)

        Q = self.output_layer(x)

        return Q


class PADDPG:
    def __init__(
            self,
            budget,
            camp,
            obs_space,
            action_space,
            epsilon_initial: float = 1.0,
            epsilon_final: float = 0.01,
            epsilon_steps: int = 1000,
            memory_size: int = 100000,
            batch_size: int = 32,
            ou_noise_theta: float = 0.15,
            ou_noise_sigma: float = 0.0001,
            gamma: float = 1,
            tau_a: float = 0.0001,
            tau_c: float = 0.0001,
            lr_a: float = 1e-3,
            lr_c: float = 1e-3,
            initial_random_steps: int = 128,
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

        """Initialization."""
        # device
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        print(self.device)

        self.obs_space = obs_space
        self.action_space = action_space

        self.num_actions = self.action_space[0].n
        self.action_parameter_sizes = np.array(
            [self.action_space.spaces[i].shape[0] for i in range(1, self.num_actions + 1)])
        self.action_parameter_size = int(self.action_parameter_sizes.sum())
        self.action_max = torch.from_numpy(np.ones((self.num_actions,))).float().to(self.device)
        self.action_min = torch.from_numpy(np.zeros((self.num_actions,))).float().to(self.device)
        self.action_range = (self.action_max - self.action_min).detach()
        self.action_parameter_max_numpy = np.concatenate(
            [self.action_space.spaces[i].high for i in range(1, self.num_actions)]).ravel()
        self.action_parameter_min_numpy = np.concatenate(
            [self.action_space.spaces[i].low for i in range(1, self.num_actions)]).ravel()
        self.action_parameter_range_numpy = (self.action_parameter_max_numpy - self.action_parameter_min_numpy)
        self.action_parameter_max = torch.from_numpy(self.action_parameter_max_numpy).float().to(self.device)
        self.action_parameter_min = torch.from_numpy(self.action_parameter_min_numpy).float().to(self.device)
        self.action_parameter_range = torch.from_numpy(self.action_parameter_range_numpy).float().to(self.device)

        self.epsilon = epsilon_initial
        self.epsilon_initial = epsilon_initial
        self.epsilon_final = epsilon_final
        self.epsilon_steps = epsilon_steps

        self.clip_grad = 40.
        self.batch_size = batch_size
        self.gamma = gamma
        self.memory_size = memory_size
        self.initial_random_steps = initial_random_steps
        self.lr_a = lr_a
        self.lr_c = lr_c
        self.tau_a = tau_a
        self.tau_c = tau_c
        self._episode = 0

        # noise
        self.noise = OUNoise(
            self.action_parameter_size,
            theta=ou_noise_theta,
            sigma=ou_noise_sigma
        )

        self.memory = ReplayBuffer(self.obs_space.shape[0], int(self.num_actions + self.action_parameter_size),
                                   memory_size, batch_size)

        # networks
        self.actor = Actor(self.obs_space.shape[0], self.num_actions, self.action_parameter_size).to(self.device)
        self.actor_target = Actor(self.obs_space.shape[0], self.num_actions, self.action_parameter_size).to(self.device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_target.eval()

        self.critic = Critic(self.obs_space.shape[0], self.num_actions, self.action_parameter_size).to(self.device)
        self.critic_target = Critic(self.obs_space.shape[0], self.num_actions, self.action_parameter_size).to(
            self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_target.eval()

        # optimizer
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=self.lr_a, betas=(0.95, 0.999))
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=self.lr_c, betas=(0.95, 0.999))

        self.writer = SummaryWriter('tensorboard-camp={}-budget={}-seed={}-{}/'.format(camp, budget, seed, time))

        # transition to store in memory
        self.transition = list()

        # total steps count
        self.total_step = 0

        # mode
        self.is_test = False

    def select_action(self, state: np.ndarray):
        """Select an action from the input state"""
        with torch.no_grad():
            x = torch.as_tensor(state, dtype=torch.float32).to(self.device)
            all_actions, all_action_parameters = self.actor.forward(x)
            all_actions = all_actions.detach().cpu().data.numpy()
            all_action_parameters = all_action_parameters.detach().cpu().data.numpy()

            if np.random.uniform() < self.epsilon:
                all_actions = np.random.uniform(size=all_actions.shape)

            action = np.argmax(all_actions)
            offset = np.array([self.action_parameter_sizes[i] for i in range(action)], dtype=int).sum()
            all_action_parameters[offset:offset + self.action_parameter_sizes[action]] += \
                self.noise.sample()[offset:offset + self.action_parameter_sizes[action]]
            all_action_parameters = np.clip(all_action_parameters, 0., 0.99)
            action_parameters = all_action_parameters[offset:offset + self.action_parameter_sizes[action]]

        self.transition = [state, list(np.concatenate((all_actions, all_action_parameters)))]
        self.total_step += 1
        return action, action_parameters, all_actions, all_action_parameters

    def store(self, reward, next_state, done):
        self.transition += [reward, next_state, done]
        self.memory.store(*self.transition)

    def update_model(self) -> Tuple:
        """Update the model by gradient descent."""
        samples = self.memory.sample_batch()
        state = torch.as_tensor(samples["obs"], dtype=torch.float32).to(self.device)
        next_state = torch.as_tensor(samples["next_obs"], dtype=torch.float32).to(self.device)
        action_combined = torch.as_tensor(samples["acts"], dtype=torch.float32).to(self.device)
        action = action_combined[:, :self.num_actions]
        action_parameters = action_combined[:, self.num_actions:]
        reward = torch.as_tensor(samples["rews"].reshape(-1, 1), dtype=torch.float32).to(self.device)
        done = torch.as_tensor(samples["done"].reshape(-1, 1), dtype=torch.float32).to(self.device)

        with torch.no_grad():
            masks = 1 - done
            next_action, next_action_parameters = self.actor_target.forward(next_state)
            next_value = self.critic_target.forward(next_state, next_action, next_action_parameters)
            curr_return = reward + self.gamma * masks * next_value

        # train critic
        values = self.critic.forward(state, action, action_parameters)
        critic_loss = F.mse_loss(values, curr_return)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        if self.clip_grad > 0:
            nn.utils.clip_grad_norm_(self.critic.parameters(), self.clip_grad)
        self.critic_optimizer.step()

        # train actor
        with torch.no_grad():
            action, action_parameters = self.actor(state)
            action_parameters = torch.cat((action, action_parameters), dim=1)

        action_parameters.requires_grad = True
        Q_val = self.critic(state, action_parameters[:, :self.num_actions],
                            action_parameters[:, self.num_actions:]).mean()
        self.critic.zero_grad()
        Q_val.backward()
        from copy import deepcopy
        delta_a = deepcopy(action_parameters.grad.data)
        action, action_parameters = self.actor(Variable(state))
        action_parameters = torch.cat((action, action_parameters), dim=1)
        delta_a[:, self.num_actions:] = self._invert_gradients(delta_a[:, self.num_actions:].cpu(),
                                                              action_parameters[:, self.num_actions:].cpu(),
                                                              grad_type="action_parameters", inplace=True)
        delta_a[:, :self.num_actions] = self._invert_gradients(delta_a[:, :self.num_actions].cpu(),
                                                              action_parameters[:, :self.num_actions].cpu(),
                                                              grad_type="actions", inplace=True)

        out = -torch.mul(delta_a, action_parameters)
        self.actor.zero_grad()
        out.backward(torch.ones(out.shape).to(self.device))
        if self.clip_grad > 0:
            nn.utils.clip_grad_norm_(self.actor.parameters(), self.clip_grad)
        self.actor_optimizer.step()

        # target update
        self._target_soft_update()

        return critic_loss.item()
    
    def update_epsilon(self):
        self._episode += 1
        
        if self._episode < self.epsilon_steps:
            self.epsilon = self.epsilon_initial - (self.epsilon_initial - self.epsilon_final) * (
                    self._episode / self.epsilon_steps)
        else:
            self.epsilon = self.epsilon_final

    def _target_soft_update(self):
        """Soft-update: target = tau*local + (1-tau)*target."""

        for t_param, l_param in zip(
                self.actor_target.parameters(), self.actor.parameters()
        ):
            t_param.data.copy_(self.tau_a * l_param.data + (1 - self.tau_a) * t_param.data)

        for t_param, l_param in zip(
                self.critic_target.parameters(), self.critic.parameters()
        ):
            t_param.data.copy_(self.tau_c * l_param.data + (1 - self.tau_c) * t_param.data)

    def _invert_gradients(self, grad, vals, grad_type, inplace=True):
        # 5x faster on CPU
        if grad_type == "actions":
            max_p = self.action_max.cpu()
            min_p = self.action_min.cpu()
            rnge = self.action_range.cpu()
        elif grad_type == "action_parameters":
            max_p = self.action_parameter_max.cpu()
            min_p = self.action_parameter_min.cpu()
            rnge = self.action_parameter_range.cpu()
        else:
            raise ValueError("Unhandled grad_type: '" + str(grad_type) + "'")

        assert grad.shape == vals.shape

        if not inplace:
            grad = grad.clone()
        with torch.no_grad():
            for n in range(grad.shape[0]):
                # index = grad < 0  # actually > but Adam minimises, so reversed (could also double negate the grad)
                index = grad[n] > 0
                grad[n][index] *= (index.float() * (max_p - vals[n]) / rnge)[index]
                grad[n][~index] *= ((~index).float() * (vals[n] - min_p) / rnge)[~index]

        return grad
