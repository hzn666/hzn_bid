import copy
import os.path
import random
from typing import Dict

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


class MultiPassQActor(nn.Module):

    def __init__(
            self,
            obs_dim: int,
            action_dim: int,
            action_parameter_dim_list: np.ndarray
    ):
        super(MultiPassQActor, self).__init__()

        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.action_parameter_dim_list = action_parameter_dim_list
        self.action_parameter_dim = sum(action_parameter_dim_list)

        self.layers = nn.Sequential(
            nn.Linear(obs_dim + self.action_parameter_dim, 128),
            nn.Linear(128, 64)
        )

        for layer in self.layers:
            nn.init.kaiming_normal_(layer.weight.data, nonlinearity="relu")
            nn.init.zeros_(layer.bias.data)

        self.action_output_layer = nn.Linear(64, action_dim)
        nn.init.normal_(self.action_output_layer.weight.data, mean=0., std=1e-5)
        nn.init.zeros_(self.action_output_layer.bias.data)

        self.offsets = self.action_parameter_dim_list.cumsum()
        self.offsets = np.insert(self.offsets, 0, 0)

    def forward(self, state, action_parameters):
        Q = []

        batch_size = state.shape[0]
        x = torch.cat((state, torch.zeros_like(action_parameters)), dim=1)
        x = x.repeat(self.action_dim, 1)

        for a in range(self.action_dim):
            x[a * batch_size:(a + 1) * batch_size,
            self.obs_dim + self.offsets[a]: self.obs_dim + self.offsets[a + 1]] = action_parameters[:,
                                                                                  self.offsets[a]:self.offsets[a + 1]]

        for layer in self.layers:
            x = F.relu(layer(x))

        Qall = self.action_output_layer(x)

        for a in range(self.action_dim):
            Qa = Qall[a * batch_size:(a + 1) * batch_size, a]
            if len(Qa.shape) == 1:
                Qa = Qa.unsqueeze(1)
            Q.append(Qa)
        Q = torch.cat(Q, dim=1)
        return Q


class QActor(nn.Module):

    def __init__(
            self,
            obs_dim: int,
            action_dim: int,
            action_parameter_dim: int,
    ):
        super(QActor, self).__init__()

        self.layers = nn.Sequential(
            nn.Linear(obs_dim + action_parameter_dim, 128),
            nn.Linear(128, 64)
        )

        for layer in self.layers:
            nn.init.kaiming_normal_(layer.weight.data, nonlinearity="relu")
            nn.init.zeros_(layer.bias.data)

        self.action_output_layer = nn.Linear(64, action_dim)
        nn.init.normal_(self.action_output_layer.weight.data, mean=0., std=1e-5)
        nn.init.zeros_(self.action_output_layer.bias.data)

    def forward(self, state, action_parameters):
        x = torch.cat((state, action_parameters), dim=1)

        for layer in self.layers:
            x = F.relu(layer(x))

        Q = self.action_output_layer(x)

        return Q


class ParamActor(nn.Module):

    def __init__(
            self,
            obs_dim,
            action_dim,
            action_parameter_dim,
    ):
        super(ParamActor, self).__init__()

        self.layers = nn.Sequential(
            nn.Linear(obs_dim, 128),
            nn.Linear(128, 64)
        )

        for layer in self.layers:
            nn.init.kaiming_normal_(layer.weight.data, nonlinearity="relu")
            nn.init.zeros_(layer.bias.data)

        self.action_parameters_output_layer = nn.Linear(64, action_parameter_dim)
        nn.init.normal_(self.action_parameters_output_layer.weight.data, mean=0., std=1e-5)
        nn.init.zeros_(self.action_parameters_output_layer.bias.data)

    def forward(self, state):
        x = state

        for layer in self.layers:
            x = F.relu(layer(x))

        action_params = self.action_parameters_output_layer(x)
        return action_params


class MPDQN:
    def __init__(
            self,
            budget,
            camp,
            obs_space,
            action_space,
            epsilon_initial: float = 1.0,
            epsilon_final: float = 0.01,
            epsilon_steps: int = 700,
            memory_size: int = 100000,
            batch_size: int = 32,
            ou_noise_theta: float = 0.15,
            ou_noise_sigma: float = 0.0001,
            gamma: float = 1,
            tau_actor: float = 0.001,
            tau_actor_param: float = 0.001,
            lr_actor: float = 0.001,
            lr_actor_param: float = 0.001,
            initial_random_steps: int = 128,
            seed: int = 1
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

        self.action_parameter_offsets = self.action_parameter_sizes.cumsum()
        self.action_parameter_offsets = np.insert(self.action_parameter_offsets, 0, 0)

        self.epsilon = epsilon_initial
        self.epsilon_initial = epsilon_initial
        self.epsilon_final = epsilon_final
        self.epsilon_steps = epsilon_steps

        self.clip_grad = 40.
        self.batch_size = batch_size
        self.gamma = gamma
        self.memory_size = memory_size
        self.initial_random_steps = initial_random_steps
        self.lr_actor = lr_actor
        self.lr_actor_param = lr_actor_param
        self.tau_actor = tau_actor
        self.tau_actor_param = tau_actor_param
        self._episode = 0
        self.invert_gradients = True

        # noise
        self.noise = OUNoise(
            self.action_parameter_size,
            theta=ou_noise_theta,
            sigma=ou_noise_sigma
        )

        self.memory = ReplayBuffer(self.obs_space.shape[0], int(1 + self.action_parameter_size),
                                   memory_size, batch_size)
        # self.actor = QActor(self.obs_space.shape[0], self.num_actions, self.action_parameter_size).to(self.device)
        # self.actor_target = QActor(self.obs_space.shape[0], self.num_actions, self.action_parameter_size).to(
        #     self.device)
        # self.actor_target.load_state_dict(self.actor.state_dict())
        # self.actor_target.eval()
        self.actor = MultiPassQActor(self.obs_space.shape[0], self.num_actions, self.action_parameter_sizes).to(
            self.device)
        self.actor_target = MultiPassQActor(self.obs_space.shape[0], self.num_actions, self.action_parameter_sizes).to(
            self.device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_target.eval()


        self.actor_param = ParamActor(self.obs_space.shape[0], self.num_actions, self.action_parameter_size).to(
            self.device)
        self.actor_param_target = ParamActor(self.obs_space.shape[0], self.num_actions, self.action_parameter_size).to(
            self.device)
        self.actor_param_target.load_state_dict(self.actor_param.state_dict())
        self.actor_param_target.eval()

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=self.lr_actor)
        self.actor_param_optimizer = optim.Adam(self.actor_param.parameters(), lr=self.lr_actor_param)

        self.writer = SummaryWriter('tensorboard/tensorboard-camp={}-budget={}-seed={}/'.format(camp, seed, budget))

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
            all_action_parameters = self.actor_param.forward(x)

            if np.random.uniform() < self.epsilon:
                action = np.random.choice(self.num_actions)
            else:
                Q_a = self.actor.forward(x.unsqueeze(0), all_action_parameters.unsqueeze(0))
                Q_a = Q_a.detach().cpu().data.numpy()
                action = np.argmax(Q_a)

            all_action_parameters = all_action_parameters.cpu().data.numpy()
            offset = np.array([self.action_parameter_sizes[i] for i in range(action)], dtype=int).sum()
            all_action_parameters[offset:offset + self.action_parameter_sizes[action]] += \
                self.noise.sample()[offset:offset + self.action_parameter_sizes[action]]
            action_parameters = all_action_parameters[offset:offset + self.action_parameter_sizes[action]]

        self.transition = [state, list(np.concatenate(([action], all_action_parameters)))]
        self.total_step += 1
        return action, action_parameters, all_action_parameters

    def store(self, reward, next_state, done):
        self.transition += [reward, next_state, done]
        self.memory.store(*self.transition)

    def update_model(self):
        """Update the model by gradient descent."""

        samples = self.memory.sample_batch()
        state = torch.as_tensor(samples["obs"], dtype=torch.float32).to(self.device)
        next_state = torch.as_tensor(samples["next_obs"], dtype=torch.float32).to(self.device)
        action_combined = torch.as_tensor(samples["acts"], dtype=torch.float32).to(self.device)
        action = action_combined[:, 0].long()
        action_parameters = action_combined[:, 1:]
        reward = torch.as_tensor(samples["rews"].reshape(-1, 1), dtype=torch.float32).to(self.device)
        done = torch.as_tensor(samples["done"].reshape(-1, 1), dtype=torch.float32).to(self.device)

        with torch.no_grad():
            masks = 1 - done
            pred_next_action_parameters = self.actor_param_target.forward(next_state)
            pred_Q_a = self.actor_target(next_state, pred_next_action_parameters)
            Q_target = torch.max(pred_Q_a, 1, keepdim=True)[0].squeeze()
            target = reward + masks * self.gamma * Q_target

        q_values = self.actor(state, action_parameters)
        Q = q_values.gather(1, action.view(-1, 1)).squeeze()
        QActor_loss = F.mse_loss(Q, Q_target)

        self.actor_optimizer.zero_grad()
        QActor_loss.backward()
        nn.utils.clip_grad_norm_(self.actor.parameters(), self.clip_grad, norm_type=2)
        self.actor_optimizer.step()

        with torch.no_grad():
            action_params = self.actor_param(state)
        action_params.requires_grad = True

        Q = self.actor(state, action_params)
        ParamActor_loss = torch.mean(torch.sum(Q, 1))
        if self.invert_gradients:
            self.actor.zero_grad()
            ParamActor_loss.backward()
            from copy import deepcopy
            delta_a = deepcopy(action_params.grad.data)

            action_params = self.actor_param(Variable(state))
            delta_a[:] = self._invert_gradients(delta_a, action_params, grad_type="action_parameters", inplace=True)

            out = -torch.mul(delta_a, action_params)
            self.actor_param.zero_grad()
            out.backward(torch.ones(out.shape).to(self.device))
            nn.utils.clip_grad_norm_(self.actor_param.parameters(), self.clip_grad, norm_type=2)
            self.actor_param_optimizer.step()
        else:
            self.actor_param_optimizer.zero_grad()
            ParamActor_loss.backward()
            self.actor_param_optimizer.step()

        # target update
        self._target_soft_update()
        return QActor_loss.item()

    def update_epsilon(self):
        self._episode += 1

        if self._episode < self.epsilon_steps:
            self.epsilon = self.epsilon_initial - (self.epsilon_initial - self.epsilon_final) * (
                    self._episode / self.epsilon_steps)
        else:
            self.epsilon = self.epsilon_final

    def save_models(self, path, prefix):
        torch.save(self.actor.state_dict(), os.path.join(path, "actor_{}.pt".format(prefix)))
        torch.save(self.actor_param.state_dict(), os.path.join(path, "actor_param_{}.pt".format(prefix)))

    def load_models(self, path):
        self.actor.load_state_dict(torch.load(os.path.join(path, "actor.pt")))
        self.actor_param.load_state_dict(torch.load(os.path.join(path, "actor_param.pt")))

    def _target_soft_update(self):
        """Soft-update: target = tau*local + (1-tau)*target."""

        for t_param, l_param in zip(
                self.actor_target.parameters(), self.actor.parameters()
        ):
            t_param.data.copy_(self.tau_actor * l_param.data + (1 - self.tau_actor) * t_param.data)

        for t_param, l_param in zip(
                self.actor_param_target.parameters(), self.actor_param.parameters()
        ):
            t_param.data.copy_(self.tau_actor_param * l_param.data + (1 - self.tau_actor_param) * t_param.data)

    def _invert_gradients(self, grad, vals, grad_type, inplace=True):
        # 5x faster on CPU
        if grad_type == "actions":
            max_p = self.action_max
            min_p = self.action_min
            rnge = self.action_range
        elif grad_type == "action_parameters":
            max_p = self.action_parameter_max
            min_p = self.action_parameter_min
            rnge = self.action_parameter_range
        else:
            raise ValueError("Unhandled grad_type: '" + str(grad_type) + "'")

        max_p = max_p.cpu()
        min_p = min_p.cpu()
        rnge = rnge.cpu()
        grad = grad.cpu()
        vals = vals.cpu()

        assert grad.shape == vals.shape

        if not inplace:
            grad = grad.clone()
        with torch.no_grad():
            # index = grad < 0  # actually > but Adam minimises, so reversed (could also double negate the grad)
            index = grad > 0
            grad[index] *= (index.float() * (max_p - vals) / rnge)[index]
            grad[~index] *= ((~index).float() * (vals - min_p) / rnge)[~index]

        return grad
