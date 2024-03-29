import numpy as np
from collections import deque, namedtuple
import copy, time, random, json, os
import torch
import torch.nn as nn
import pickle

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class ConfigurableObject:
    """
    A base class of all objects that could be configured with a flattened dict.
    The values could be objects or functions/lambdas.
    """
    def __init__(self, config, default={}):
        self.config = copy.deepcopy(config)
        self.default = copy.deepcopy(default)

    def has(self, k):
        return k in self.config or k in self.default

    def get(self, k):
        if k in self.config:
            return self.config[k]
        elif k in self.default:
            return self.default[k]
        else:
            return None

    def print_config(self):
        config = self.default
        for k in self.config:
            config[k] = self.config[k]
        print('Configs: ', json.dumps(config, indent=4))


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
    def __init__(self, window_size=100, log_file=None, log_interval=50):
        self.reward_log = SmoothAccumulator(window_size)
        self.time_log = SmoothAccumulator(window_size)
        self._episode_count = 0
        self._start_timestamp = None
        self._log_file = log_file
        self._log_interval = log_interval

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
        if self._episode_count % self._log_interval == 0:
            print('')
            if self._log_file:
                with open(self._log_file, 'wb') as f:
                    pickle.dump(self, f)

    def get_all_rewards(self):
        return self.reward_log.get_all_records()


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
    Based on the implementation in Udacity DDPG mini project:
    https://github.com/udacity/deep-reinforcement-learning/blob/d6cb43c1b11b1d55c13ac86d6002137c7b880c15/ddpg-pendulum/ddpg_agent.py
    """

    def __init__(self, size, seed, mu=0., theta=0.15, sigma=0.2, discount=1.0):
        """Initialize parameters and noise process."""
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        self.seed = random.seed(seed)
        self.discount = discount
        self.reset()

    def reset(self):
        """Reset the internal state (= noise) to mean (mu)."""
        self.state = copy.copy(self.mu)
        self.coef = 1.
        # The first several samples are highly biased so we skip them.
        for _ in range(10):
            self.sample()

    def sample(self):
        """Update internal state and return it as a noise sample."""
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.array([random.random() for i in range(len(x))])
        self.state = x + dx
        self.coef *= self.discount
        return self.state * self.coef


def TrainDDPG(env, agent, config):
    
    # Copy the configs.
    action_repeat = config['action_repeat']
    action_size = config['action_size']
    state_size = config['state_size'] * action_repeat
    seed = config['seed']
    batch_num = config['batch_num']
    max_step_num = config['max_step_num']
    max_episode_num = config['max_episode_num']
    learn_interval = config['learn_interval']
    out_low = config['out_low']
    out_high = config['out_high']
    
    # Prepare the utilities.
    os.makedirs(config['model_dir'], exist_ok=True)
    buffer = ReplayBuffer(config['buffer_size'], batch_num, seed)
    noise = OUNoise(action_size, seed, discount=config['noise_discount'])
    logger = RLTrainingLogger(config['window_size'], config['log_file'], config['log_interval'])
    
    # The main training process.
    total_step_num = 0
    total_episode_num = 0
    
    get_last_repeated_state = lambda states: np.concatenate(states[-action_repeat:], 1)
    
    best_performance = None
    
    while total_step_num < max_step_num and total_episode_num < max_episode_num:
        # Start an episode
        raw_state = env.reset()
        done = False
        logger.episode_begin()
        state = np.concatenate([raw_state for _ in range(action_repeat)], 1)
        episode_rewards = []
        while not done:
            action = agent.act(state)
            action += noise.sample()
            action = np.clip(action, a_min=out_low, a_max=out_high)
            states = []
            for _ in range(action_repeat):
                next_raw_state, reward, done = env.step(action)
                total_step_num += 1
                states.append(next_raw_state)
                episode_rewards.append(reward)
            # print('Interacting: next_state.shape =', next_state.shape)
            next_state = np.concatenate(states, 1)
            buffer.add(state, action, sum(episode_rewards[-action_repeat:]), next_state, done)
            state = next_state
            if len(buffer) > batch_num and total_step_num % learn_interval == 0:
                agent.learn(buffer.sample())
        logger.episode_end(sum(episode_rewards))
        total_episode_num += 1
        
        # Save the model as long as its recent smoothed reward is higher than
        # the previous best performance by some margin.
        smooth_performance = logger.reward_log.get_latest_record()[1]
        if best_performance is None or smooth_performance > best_performance + .5:
            agent.save_model(config.get('model_dir'))
            best_performance = smooth_performance
