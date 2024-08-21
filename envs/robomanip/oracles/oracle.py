import numpy as np


class Oracle:
    def __init__(self, env, min_norm=0.3):
        self._env = env
        self._min_norm = min_norm
        self._debug = True
        self._done = False

        if self._debug:
            np.set_printoptions(suppress=True)

    def shape_diff(self, diff):
        diff_norm = np.linalg.norm(diff)
        if diff_norm >= self._min_norm:
            return diff
        else:
            return diff / (diff_norm + 1e-6) * self._min_norm

    def shortest_yaw(self, eff_yaw, obj_yaw, n=4):
        symmetries = np.array([i * 2 * np.pi / n + obj_yaw for i in range(-n, n + 1)])
        d = np.argmin(np.abs(eff_yaw - symmetries))
        return symmetries[d]

    def print_phase(self, phase):
        if self._debug:
            print(f'Phase {phase:50}', end=' ')

    @property
    def done(self):
        return self._done

    def reset(self, ob, info):
        pass

    def select_action(self, ob, info):
        pass
