import os

import mujoco_py
import numpy as np
from gym import utils
from gym.envs.mujoco import mujoco_env

GYM_ASSETS_DIR = os.path.join(os.path.dirname(__file__), 'assets')


class AntEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    """Basic ant locomotion environment."""
    FILE = os.path.join(GYM_ASSETS_DIR, 'ant.xml')

    def __init__(self, file_path=None, expose_all_qpos=False,
                 expose_body_coms=None, expose_body_comvels=None, non_zero_reset=False):
        if file_path is None:
            file_path = self.FILE

        self._expose_all_qpos = expose_all_qpos
        self._expose_body_coms = expose_body_coms
        self._expose_body_comvels = expose_body_comvels
        self._body_com_indices = {}
        self._body_comvel_indices = {}

        self._non_zero_reset = non_zero_reset

        mujoco_env.MujocoEnv.__init__(self, file_path, 5)
        utils.EzPickle.__init__(self)

    @property
    def physics(self):
        # Check mujoco version is greater than version 1.50 to call correct physics
        # model containing PyMjData object for getting and setting position/velocity.
        # Check https://github.com/openai/mujoco-py/issues/80 for updates to api.
        if mujoco_py.get_version() >= '1.50':
            return self.sim
        else:
            return self.model

    def _step(self, a):
        return self.step(a)

    def step(self, a):
        xposbefore = self.get_body_com("torso")[0]
        self.do_simulation(a, self.frame_skip)
        xposafter = self.get_body_com("torso")[0]
        forward_reward = (xposafter - xposbefore) / self.dt
        ctrl_cost = .5 * np.square(a / 30).sum()
        contact_cost = 0.5 * 1e-3 * np.sum(
            np.square(np.clip(self.sim.data.cfrc_ext, -1, 1)))
        survive_reward = 1.0
        reward = forward_reward - ctrl_cost - contact_cost + survive_reward
        state = self.state_vector()
        notdone = np.isfinite(state).all() \
                  and state[2] >= 0.2 and state[2] <= 1.0
        done = not notdone
        ob = self._get_obs()
        return ob, reward, done, dict(
            reward_forward=forward_reward,
            reward_ctrl=-ctrl_cost,
            reward_contact=-contact_cost,
            reward_survive=survive_reward,
            reward=reward,
        )

    def _get_obs(self):
        # No cfrc observation.
        if self._expose_all_qpos:
            obs = np.concatenate([
                self.physics.data.qpos.flat[:15],  # Ensures only ant obs.
                self.physics.data.qvel.flat[:14],
            ])
        else:
            obs = np.concatenate([
                self.physics.data.qpos.flat[2:15],
                self.physics.data.qvel.flat[:14],
            ])

        if self._expose_body_coms is not None:
            for name in self._expose_body_coms:
                com = self.get_body_com(name)
                if name not in self._body_com_indices:
                    indices = range(len(obs), len(obs) + len(com))
                    self._body_com_indices[name] = indices
                obs = np.concatenate([obs, com])

        if self._expose_body_comvels is not None:
            for name in self._expose_body_comvels:
                comvel = self.get_body_comvel(name)
                if name not in self._body_comvel_indices:
                    indices = range(len(obs), len(obs) + len(comvel))
                    self._body_comvel_indices[name] = indices
                obs = np.concatenate([obs, comvel])
        return obs

    def reset_model(self):
        qpos = self.init_qpos + self.np_random.uniform(
            size=self.model.nq, low=-.1, high=.1)
        qvel = self.init_qvel + self.np_random.randn(self.model.nv) * .1

        if self._non_zero_reset:
            """Now the reset is supposed to be to a non-zero location"""
            reset_location = self._get_reset_location()
            qpos[:2] = reset_location

        # Set everything other than ant to original position and 0 velocity.
        qpos[15:] = self.init_qpos[15:]
        qvel[14:] = 0.
        self.set_state(qpos, qvel)
        return self._get_obs()

    def viewer_setup(self):
        self.viewer.cam.distance = self.model.stat.extent * 0.5

    def get_xy(self):
        return self.physics.data.qpos[:2]

    def set_xy(self, xy):
        qpos = np.copy(self.physics.data.qpos)
        qpos[0] = xy[0]
        qpos[1] = xy[1]
        qvel = self.physics.data.qvel
        self.set_state(qpos, qvel)

    def render(self, *args, **kwargs):
        frame = super().render(mode='rgb_array', width=200, height=200).transpose(2, 0, 1)
        return frame
