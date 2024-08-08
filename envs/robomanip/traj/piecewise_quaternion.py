from __future__ import annotations

import numpy as np

from envs.robomanip.lie import interpolate
from envs.robomanip.traj.piecewise_trajectory import PiecewiseTrajectory


class PiecewiseQuaternionSlerp(PiecewiseTrajectory):
    """A class representing a trajectory for quaternions that are interpolated using
    piecewise spherical linear interpolation (slerp).
    """

    def __init__(self, breaks, samples):
        super().__init__(breaks, samples)

    def value(self, t):
        s = self.get_segment_index(t)
        interp_time = (t - self.start_time(s)) / self.duration(s)
        interp_time = np.clip(interp_time, 0.0, 1.0)
        return interpolate(self.samples[s], self.samples[s + 1], interp_time)
