import numpy as np
from gym import Wrapper
from gym.envs.mujoco.mujoco_env import convert_observation_to_space


class XYWrapper(Wrapper):
    def __init__(self, env, resample_interval=100):
        super().__init__(env)

        self.z = None
        self.num_steps = 0
        self.resample_interval = resample_interval

        ob = self.reset()
        self.observation_space = convert_observation_to_space(ob)

    def reset(self, task_idx=None):
        ob = self.env.reset()
        self.z = np.random.randn(2)
        self.z = self.z / np.linalg.norm(self.z)
        self.num_steps = 0

        return np.concatenate([ob, self.z])

    def step(self, action):
        cur_xy = self.env.get_xy().copy()
        ob, reward, done, info = self.env.step(action)
        next_xy = self.env.get_xy().copy()
        self.num_steps += 1

        reward = (next_xy - cur_xy).dot(self.z)
        info['xy'] = cur_xy
        info['direction'] = self.z

        if self.num_steps % self.resample_interval == 0:
            self.z = np.random.randn(2)
            self.z = self.z / np.linalg.norm(self.z)

        return np.concatenate([ob, self.z]), reward, done, info
