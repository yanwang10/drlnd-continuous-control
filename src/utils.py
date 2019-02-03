import numpy as np
from collections import deque, namedtuple
import copy, time, random
import torch
import torch.nn as nn

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class SmoothAccumulator:
    """
    An accumulator that records the raw values and smoothed values within a
    certain window.
    """
    def __init__(self, window_size):
        self.window_size = window_size
        self._raw = list()
        self._smooth = list()
        self._window = deque(maxlen=self.window_size)

    def add(self, v):
        self._raw.append(v)
        self._window.append(v)
        self._smooth.append(np.mean(self._window))

    def get_latest_record(self):
        return self._raw[-1], self._smooth[-1]

    def get_all_records(self):
        return self._raw, self._smooth


class RLTrainingLogger:
    """
    A helper class that logs the rewards and running time of training process.
    """
    def __init__(self, window_size=100):
        self.reward_log = SmoothAccumulator(window_size)
        self.time_log = SmoothAccumulator(window_size)
        self._episode_count = 0
        self._start_timestamp = None

    def episode_begin(self):
        self._start_timestamp = time.time()

    def episode_end(self, reward):
        time_interval = time.time() - self._start_timestamp
        self._start_timestamp = None
        self.reward_log.add(reward)
        self.time_log.add(time_interval)
        self._episode_count += 1
        r, smooth_r = self.reward_log.get_latest_record()
        t, smooth_t = self.time_log.get_latest_record()
        print('\rEpisode %4d: reward = %.3f (%.3f), time = %.2fs (%.2fs)' %
              (self._episode_count, r, smooth_r, t, smooth_t), end='')

    def get_smooth_rewards(self):
        return self.reward_log.get_all_records()[1]


class ReplayBuffer:
    """
    Fixed-size buffer to store experience tuples.
    Copied from the Udacity DQN mini project.
    """

    def __init__(self, buffer_size, batch_size, seed):
        """Initialize a ReplayBuffer object.
        Params
        ======
            action_size (int): dimension of each action
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
            seed (int): random seed
        """
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.seed = random.seed(seed)

    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)

    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=self.batch_size)

        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).float().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)

        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)


class OUNoise:
    """
    Ornstein-Uhlenbeck process.
    Copied from Udacity DDPG mini project:
    https://github.com/udacity/deep-reinforcement-learning/blob/d6cb43c1b11b1d55c13ac86d6002137c7b880c15/ddpg-pendulum/ddpg_agent.py
    """

    def __init__(self, size, seed, mu=0., theta=0.15, sigma=0.2):
        """Initialize parameters and noise process."""
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        self.seed = random.seed(seed)
        self.reset()

    def reset(self):
        """Reset the internal state (= noise) to mean (mu)."""
        self.state = copy.copy(self.mu)
        # The first several samples are highly biased so we skip them.
        for _ in range(10):
            self.sample()

    def sample(self):
        """Update internal state and return it as a noise sample."""
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.array([random.random() for i in range(len(x))])
        self.state = x + dx
        return self.state

def layer_init(layer, w_scale=0.1):
    """
    Initialize the parameters (weights and biases) of a fully-connected layer.
    Copied from https://github.com/ShangtongZhang/DeepRL/blob/3437ddfbe0ca7e4d57f4753f959f01d34091794e/deep_rl/network/network_utils.py
    """
    nn.init.orthogonal_(layer.weight.data)
    layer.weight.data.mul_(w_scale)
    nn.init.constant_(layer.bias.data, 0)
    return layer
