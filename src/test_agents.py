import gym
import numpy as np
from env_wrapper import EnvWrapper
from ddpg import DDPGAgent
from utils import RLTrainingLogger

pendulum = gym.make('Pendulum-v0')
env = EnvWrapper(gym_env=pendulum, max_episode_len=300)
config = {
    'state_size': pendulum.observation_space.shape[0],
    'action_size': 1,
    'tau': 1e-2,
    'actor_lr': 1e-2,
    'critic_lr': 1e-2,
}
agent = DDPGAgent(config)

max_episode = 1000
logger = RLTrainingLogger(window_size=100)
renderable = True
pendulum.reset()
try:
    pendulum.render()
except:
    renderable = False

for _ in range(max_episode):
    state = env.reset()
    done = False
    rewards = 0.0
    logger.episode_begin()
    actions = []
    while not done:
        if renderable:
            pendulum.render()
        action = agent.act(state)
        if len(actions) == 10:
            # print(' '.join(actions))
            actions = list()
        actions.append('%.2f' % action)
        next_state, reward, done = env.step(action * 2)
        rewards += reward
        agent.step(state, action, reward, next_state, done)
        state = next_state
    logger.episode_end(rewards)
    agent.reset()

