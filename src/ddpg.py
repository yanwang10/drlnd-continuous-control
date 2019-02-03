import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from .base_agent import *
from .utils import *

DEFAULT_CONFIG = {
    'state_size': None,
    'action_size': None,
    'seed': 123456,
    'batch_size': 128,
    'buffer': int(1e5),
    'tau': 1e-3,
    'actor_lr': 1e-4,
    'critic_lr': 1e-3,
    'gamma': 0.99,
}

def MergeToDict(src, dst):
    for k, v in src.iteritems():
        if not k in dst:
            dst[k] = v
    return dst

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class FCNetwork(nn.Module):
    def __init__(self, input_size, output_size, seed, hidden):
        super(FCNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        dims = (input_size, ) + hidden + (output_size, )
        layers = list()
        for i in range(1, len(dims)):
            layers.append(nn.Linear(dims[i - 1], dims[i]))
        self.layers = nn.ModuleList(layers)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class MuNetwork(nn.Module):
    def __init__(self, state_size, action_size, seed, hidden=(16, 16, 16)):
        super(MuNetwork, self).__init__()
        dims = (state_size, ) + hidden + (action_size, )
        layers = list()
        for i in range(1, len(dims)):
            layers.append(nn.Linear(dims[i - 1], dims[i]))
        self.layers = nn.ModuleList(layers)

    def forward(self, state):
        for i, layer in enumerate(self.layers):
            state = layer(state)
            if i == len(self.layers) - 1:
                state = F.tanh(state)
            else:
                state = F.relu(state)
        return state


class QNetwork(nn.Module):
    def __init__(self, state_size, action_size, seed, hidden=(32, 32, 32)):
        super(QNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)

        # Then concatenate with action and go through other fully connected
        # layers.
        dims = (state_size + action_size, ) + hidden + (1, )
        layers = []
        for i in range(1, len(dims)):
            layers.append(layer_init(nn.Linear(dims[i - 1], dims[i])))
        self.layers = nn.ModuleList(layers)

    def forward(self, state, action):
        x = torch.cat([state, action], dim=1)
        for layer in self.layers:
            x = layer(F.relu(x))
        return x

class DDPGAgent(BaseAgent):
    """
    Implementation of DDPG algorithm described in this paper:
    Continuous Control with Deep Reinforcement Learning
    https://arxiv.org/pdf/1509.02971.pdf
    """
    def __init__(self, config):
        super(DDPGAgent, self).__init__(config, DEFAULT_CONFIG)
        state_size = self.get('state_size')
        action_size = self.get('action_size')
        seed = self.get('seed')
        self.q_local = QNetwork(state_size, action_size, seed).to(device)
        self.q_optimizer = optim.Adam(
            self.q_local.parameters(), lr=self.get('actor_lr'))
        self.mu_local = MuNetwork(state_size, action_size, seed).to(device)
        self.mu_optimizer = optim.Adam(
            self.mu_local.parameters(), lr=self.get('critic_lr'))

        # For some unknown reason copy.deepcopy fails to copy a model with error
        # message: TypeError: can't pickle torch._C.Generator objects
        # So we have to create new networks with the same structure and copy the
        # weights.
        self.q_target = QNetwork(state_size, action_size, seed).to(device)
        self.q_target.load_state_dict(self.q_local.state_dict())
        self.mu_target = MuNetwork(state_size, action_size, seed).to(device)
        self.mu_target.load_state_dict(self.mu_local.state_dict())
        self.buffer = ReplayBuffer(
            self.get('buffer_size'), self.get('batch_size'), seed)

        self.noise = OUNoise(action_size, seed)

        self.print_config()
        self.episode_num = 1

    def reset(self):
        self.noise.reset()
        self.episode_num += 1

    def act(self, state):
        if type(state) == dict:
            state = state.get('ReacherBrain')
        state = torch.from_numpy(state).float().to(device)
        self.mu_local.eval()
        with torch.no_grad():
            action = self.mu_local(state).cpu().data.numpy()
        self.mu_local.train()
        if self.episode_num <= 100:
            action += self.noise.sample() * 1.0
        return np.clip(action, -1., 1.)

    def step(self, state, action, reward, next_state, done):
        self.buffer.add(state, action, reward, next_state, done)
        if len(self.buffer) > self.get('batch_size'):
            experiences = self.buffer.sample()
            states, actions, rewards, next_states, dones = experiences

            # Update the critic.
            mu_prime = self.mu_target(next_states)
            q_prime = self.q_target(next_states, mu_prime)
            y = rewards + self.get('gamma') * q_prime * (1. - dones)
            y = y.detach()
            q = self.q_local(states, actions)
            critic_loss = F.mse_loss(y, q)
            self.q_optimizer.zero_grad()
            critic_loss.backward()
            self.q_optimizer.step()

            # Update the actor.
            predicted_actions = self.mu_local(states)
            policy_loss = -self.q_local(states, predicted_actions).mean()
            self.mu_optimizer.zero_grad()
            policy_loss.backward()
            self.mu_optimizer.step()

            # Soft update the target networks.
            self.soft_update(self.q_local, self.q_target)
            self.soft_update(self.mu_local, self.mu_target)


    def soft_update(self, src, dst):
        tau = self.get('tau')
        for dst_param, src_param in zip(dst.parameters(), src.parameters()):
            dst_param.detach_()
            dst_param.data.copy_(
                tau * src_param.data + (1.0 - tau) * dst_param.data)
