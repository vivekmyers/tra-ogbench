import gymnasium
import numpy as np
from crafter import Env, constants
from gymnasium.spaces import Box, Discrete


class CrafterEnv(gymnasium.Env):
    metadata = {
        'render_modes': ['rgb_array'],
        'render_fps': 15,
    }

    @property
    def observation_space(self):
        return Box(0, 255, tuple(self.env._size) + (3,), np.uint8)

    @property
    def action_space(self):
        return Discrete(len(constants.actions))

    @property
    def action_names(self):
        return constants.actions

    def __init__(self, *args, **kwargs):
        self.env = Env(*args, **kwargs)

    def reset(self, *, seed=None, options=None):
        ob = self.env.reset()
        info = dict()
        return ob, info

    def step(self, action):
        ob, reward, done, info = self.env.step(action)

        terminated = self.env._player.health <= 0
        truncated = self.env._length and self.env._step >= self.env._length
        assert done == (terminated or truncated)

        return ob, reward, terminated, truncated, info

    def render(self):
        return self.env.render()
