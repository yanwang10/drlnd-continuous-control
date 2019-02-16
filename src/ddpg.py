import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from networks import *
from torchsummary import summary

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class DDPGAgent:
    """
    Implementation of DDPG algorithm described in this paper:
    Continuous Control with Deep Reinforcement Learning
    https://arxiv.org/pdf/1509.02971.pdf
    """
    def __init__(self, config):
        super(DDPGAgent, self).__init__()
        self.tau = config.get('tau')
        self.gamma = config.get('gamma')
        action_repeat = config.get('action_repeat')
        state_size = config.get('state_size') * action_repeat
        action_range = [config.get('out_low'), config.get('out_high')]
        action_size = config.get('action_size')
        seed = config.get('seed')
        actor_hidden = config.get('actor_hidden')
        critic_hidden = config.get('critic_hidden')
        self.q_local = CriticNetwork(
            state_size + action_size, action_size, critic_hidden, seed=seed).to(device)
        summary(self.q_local, (state_size + action_size,))
        self.q_optimizer = optim.Adam(self.q_local.parameters(), lr=config.get('critic_lr'))
        self.mu_local = ActorNetwork(
            state_size, 1, actor_hidden, action_range, seed).to(device)
        summary(self.mu_local, (state_size,))
        self.mu_optimizer = optim.Adam(self.mu_local.parameters(), lr=config.get('actor_lr'))

        # For some unknown reason copy.deepcopy fails to copy a model with error
        # message: TypeError: can't pickle torch._C.Generator objects
        # So we have to create new networks with the same structure and copy the
        # weights.
        self.q_target = CriticNetwork(
            state_size + action_size, 1, critic_hidden, seed).to(device)
        self.q_target.load_state_dict(self.q_local.state_dict())
        self.mu_target = ActorNetwork(
            state_size, action_size, actor_hidden,
            out_range=action_range, seed=seed).to(device)
        self.mu_target.load_state_dict(self.mu_local.state_dict())

    def act(self, state):
        if type(state) == dict:
            state = state.get('ReacherBrain')
        state = torch.from_numpy(state).float().to(device)
        self.mu_local.eval()
        with torch.no_grad():
            action = self.mu_local(state).cpu().data.numpy()
        self.mu_local.train()
        return action

    def learn(self, experiences):
        states, actions, rewards, next_states, dones = experiences
        # Update the critic.
        mu_prime = self.mu_target(next_states)
        q_prime = self.q_target(torch.cat((next_states, mu_prime), 1))
        y = rewards + self.gamma * q_prime * (1. - dones)
        y = y.detach()
        q = self.q_local(torch.cat((states, actions), 1))
        critic_loss = F.mse_loss(y, q)
        self.q_optimizer.zero_grad()
        critic_loss.backward()
        self.q_optimizer.step()

        # Update the actor.
        predicted_actions = self.mu_local(states)
        policy_loss = -self.q_local(torch.cat((states, predicted_actions), 1)).mean()
        self.mu_optimizer.zero_grad()
        policy_loss.backward()
        self.mu_optimizer.step()

        # Soft update the target networks.
        self.soft_update(self.q_local, self.q_target)
        self.soft_update(self.mu_local, self.mu_target)


    def soft_update(self, src, dst):
        for dst_param, src_param in zip(dst.parameters(), src.parameters()):
            dst_param.detach_()
            dst_param.data.copy_(
                self.tau * src_param.data + (1.0 - self.tau) * dst_param.data)
