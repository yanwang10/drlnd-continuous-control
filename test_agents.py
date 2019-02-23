import gym
import json
import numpy as np
from src import *

np.set_printoptions(precision=3)

pendulum = gym.make('Pendulum-v0')
env = EnvWrapper(gym_env=pendulum, max_episode_len=300)

"""
With these settings on the Pendulum-v0 environment, the DDPG agent
gets >-250 reward in a window of 100 episode after 800 training
episodes.
"""

config = {
    # Configs about the env.
    'state_size': pendulum.observation_space.shape[0],
    'action_size': pendulum.action_space.shape[0],
    'out_low': np.asscalar(pendulum.action_space.low),
    'out_high': np.asscalar(pendulum.action_space.high),
    
    # Configs for the agent.
    'tau': 1e-3,
    'gamma': 0.96,
    'init_weight_scale': 0.3,
    'grad_clip': 10.,
    'actor_hidden': [64, 64, 64],
    'actor_lr': 1e-3,
    'critic_hidden': [128, 128, 128],
    'critic_lr': 1e-4,
    'action_repeat': 3,
    
    # Configs for the training process.
    'noise_discount': 0.9997,
    'seed': 1317317,
    'buffer_size': 1000 * 300,
    'batch_num': 16,
    'model_dir': './saved_model',
    'max_episode_num': 3000,
    'max_step_num': 3000 * 300,
    'learn_interval': 1,
    
    # Configs for logging.
    'log_dir': './logs',
    'window_size': 100,
}
print(json.dumps(config, indent=4))
agent = DDPGAgent(config)
TrainDDPG(env, agent, config)
