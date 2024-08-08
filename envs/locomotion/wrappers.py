import gymnasium
import numpy as np
from gymnasium.spaces import Box


class GymXYWrapper(gymnasium.Wrapper):
    def __init__(self, env, resample_interval=100):
        super().__init__(env)

        self.z = None
        self.num_steps = 0
        self.resample_interval = resample_interval

        ob, _ = self.reset()
        self.observation_space = Box(low=-np.inf, high=np.inf, shape=ob.shape, dtype=np.float64)

    def reset(self, *args, **kwargs):
        ob, info = self.env.reset(*args, **kwargs)
        self.z = np.random.randn(2)
        self.z = self.z / np.linalg.norm(self.z)
        self.num_steps = 0

        return np.concatenate([ob, self.z]), info

    def step(self, action):
        cur_xy = self.unwrapped.data.qpos[:2].copy()
        ob, reward, terminated, truncated, info = self.env.step(action)
        next_xy = self.unwrapped.data.qpos[:2].copy()
        self.num_steps += 1

        reward = (next_xy - cur_xy).dot(self.z)
        info['xy'] = next_xy
        info['direction'] = self.z

        if self.num_steps % self.resample_interval == 0:
            self.z = np.random.randn(2)
            self.z = self.z / np.linalg.norm(self.z)

        return np.concatenate([ob, self.z]), reward, terminated, truncated, info


class DMCHumanoidXYWrapper(GymXYWrapper):
    def step(self, action):
        from envs.locomotion.humanoid import tolerance

        cur_xy = self.unwrapped.data.qpos[:2].copy()
        ob, reward, terminated, truncated, info = self.env.step(action)
        next_xy = self.unwrapped.data.qpos[:2].copy()
        self.num_steps += 1

        head_height = self.unwrapped.data.xpos[2, 2]  # ['head', 'z']
        torso_upright = self.unwrapped.data.xmat[1, 8]  # ['torso', 'zz']
        center_of_mass_velocity = self.unwrapped.data.sensordata[0:3]  # ['torso_subtreelinvel']
        control = self.unwrapped.data.ctrl.copy()

        standing = tolerance(head_height, bounds=(1.4, float('inf')), margin=1.4 / 4)
        upright = tolerance(torso_upright, bounds=(0.9, float('inf')), margin=1.9, sigmoid='linear', value_at_margin=0)
        stand_reward = standing * upright
        small_control = tolerance(control, margin=1, value_at_margin=0, sigmoid='quadratic').mean()
        small_control = (4 + small_control) / 5
        move = center_of_mass_velocity[0:2].dot(self.z)
        move = (5 * move + 1) / 6
        # reward = small_control * stand_reward * move
        reward = stand_reward * (1 + (next_xy - cur_xy).dot(self.z) * 100)

        info['xy'] = next_xy
        info['direction'] = self.z

        if self.num_steps % self.resample_interval == 0:
            self.z = np.random.randn(2)
            self.z = self.z / np.linalg.norm(self.z)

        return np.concatenate([ob, self.z]), reward, terminated, truncated, info
