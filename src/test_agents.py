import gym
import json
import numpy as np
from env_wrapper import EnvWrapper
from ddpg import DDPGAgent
from utils import *

pendulum = gym.make('Pendulum-v0')
env = EnvWrapper(gym_env=pendulum, max_episode_len=300)
config = {
    # Configs about the env.
    'state_size': pendulum.observation_space.shape[0],
    'action_size': pendulum.action_space.shape[0],
    'out_low': np.asscalar(pendulum.action_space.low),
    'out_high': np.asscalar(pendulum.action_space.high),
    
    # Configs for the agent.
    'tau': 1e-2,
    'gamma': 0.95,
    'actor_hidden': [8, 8],
    'actor_lr': 1e-2,
    'critic_hidden': [16, 16],
    'critic_lr': 1e-2,
    
    # Configs for the training process.
    'noise_discount': 0.9996,
    'seed': 1317317,
    'buffer_size': 1000 * 300,
    'batch_num': 64,
    'model_dir': './saved_model',
    'max_episode_num': 2000,
    'max_step_num': 2000 * 300,
    'learn_interval': 16,
    
    # Configs for logging.
    'log_dir': './logs',
    'window_size': 100,
}
print(json.dumps(config, indent=4))
agent = DDPGAgent(config)
Train(env, agent, config)
