import os

import numpy as np
from gymnasium import utils
from gymnasium.envs.mujoco import MujocoEnv
from gymnasium.spaces import Box


# TODO: Implement
class HumanoidEnv(MujocoEnv, utils.EzPickle):
    xml_file = os.path.join(os.path.dirname(__file__), 'assets', 'humanoid.xml')
    metadata = {
        'render_modes': ['human', 'rgb_array', 'depth_array'],
        'render_fps': 40,
    }

    def __init__(
            self,
            xml_file=None,
            reset_noise_scale=0.1,
            **kwargs,
    ):
        if xml_file is None:
            xml_file = self.xml_file
        utils.EzPickle.__init__(
            self,
            xml_file,
            reset_noise_scale,
            **kwargs,
        )

        self._reset_noise_scale = reset_noise_scale

        obs_shape = 29

        observation_space = Box(low=-np.inf, high=np.inf, shape=(obs_shape,), dtype=np.float64)

        MujocoEnv.__init__(
            self,
            xml_file,
            frame_skip=5,
            observation_space=observation_space,
            default_camera_config={
                'distance': 4.0,
            },
            **kwargs,
        )

    @property
    def terminated(self):
        terminated = False
        return terminated

    def step(self, action):
        self.do_simulation(action, self.frame_skip)

        terminated = self.terminated
        observation = self._get_obs()

        if self.render_mode == 'human':
            self.render()

        return observation, 0., terminated, False, {
            'xy': self.get_xy(),
        }

    def _get_obs(self):
        position = self.data.qpos.flat.copy()
        velocity = self.data.qvel.flat.copy()

        return np.concatenate([position, velocity])

    def reset_model(self):
        noise_low = -self._reset_noise_scale
        noise_high = self._reset_noise_scale

        qpos = self.init_qpos + self.np_random.uniform(low=noise_low, high=noise_high, size=self.model.nq)
        qvel = self.init_qvel + self._reset_noise_scale * self.np_random.standard_normal(self.model.nv)
        self.set_state(qpos, qvel)

        observation = self._get_obs()

        return observation

    def get_xy(self):
        return self.data.qpos[:2]

    def set_xy(self, xy):
        qpos = self.data.qpos.copy()
        qvel = self.data.qvel.copy()
        qpos[:2] = xy
        self.set_state(qpos, qvel)
