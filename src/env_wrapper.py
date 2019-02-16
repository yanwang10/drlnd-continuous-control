import numpy as np

GYM = 'gym'
UNITY = 'unity'

class EnvWrapper:
    """
    The wrapper that wraps environment from openai gym or unity ml-agent.
    """
    def __init__(self, gym_env=None, unity_env=None,
                 brain_name=None, max_episode_len=1000):
        self.max_episode_len = max_episode_len
        self.this_episode_len = 0
        self.env = None
        if gym_env:
            print('Using gym env')
            self.env = gym_env
            self.env_type = GYM
        elif unity_env:
            print('Using unity env with brain name "%s"' % brain_name)
            self.env = unity_env
            self.brain_name = brain_name
            self.env_type = UNITY
        else:
            raise NotImplementedError

    def transform_obs(self, obs):
        if self.env_type == GYM:
            if len(obs.shape) == 1:
                return np.expand_dims(obs, axis=0)
            obs = obs.reshape(1, -1)
            return obs
        elif self.env_type == UNITY:
            return obs
        else:
            raise NotImplementedError

    def reset(self):
        self.this_episode_len = 0
        if self.env_type == GYM:
            return self.transform_obs(self.env.reset())
        elif self.env_type == UNITY:
            env_info = self.env.reset(train_mode=True)[self.brain_name]
            return env_info.vector_observations[0]
        else:
            raise NotImplementedError

    def step(self, action):
        self.this_episode_len += 1
        if self.env_type == GYM:
            obs, reward, done, _ = self.env.step(action)
            obs = self.transform_obs(obs)
            done = done | self.this_episode_len >= self.max_episode_len
            return obs, reward, done
        elif self.env_type == UNITY:
            env_info = self.env.step(action)[self.brain_name]
            obs = env_info.vector_observations[0]
            reward = env_info.rewards[0]
            done = env_info.local_done[0]
            obs = self.transform_obs(obs)
            done = done | self.this_episode_len >= self.max_episode_len
            return obs, reward, done
        else:
            raise NotImplementedError
