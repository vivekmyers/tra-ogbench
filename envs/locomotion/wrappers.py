import numpy as np
import gymnasium
from gymnasium.spaces import Box


class XYWrapper(gymnasium.Wrapper):
    def __init__(self, env, resample_interval=100):
        super().__init__(env)

        self.z = None
        self.num_steps = 0
        self.resample_interval = resample_interval

        ob, _ = self.reset()
        self.observation_space = Box(
            low=-np.inf, high=np.inf, shape=ob.shape, dtype=np.float64
        )

    def reset(self, *args, **kwargs):
        ob, info = self.env.reset(*args, **kwargs)
        self.z = np.random.randn(2)
        self.z = self.z / np.linalg.norm(self.z)
        self.num_steps = 0

        return np.concatenate([ob, self.z]), info

    def step(self, action):
        cur_xy = self.env.data.qpos[:2].copy()
        ob, reward, terminated, truncated, info = self.env.step(action)
        next_xy = self.env.data.qpos[:2].copy()
        self.num_steps += 1

        reward = (next_xy - cur_xy).dot(self.z)
        info['xy'] = cur_xy
        info['direction'] = self.z

        if self.num_steps % self.resample_interval == 0:
            self.z = np.random.randn(2)
            self.z = self.z / np.linalg.norm(self.z)

        return np.concatenate([ob, self.z]), reward, terminated, truncated, info
