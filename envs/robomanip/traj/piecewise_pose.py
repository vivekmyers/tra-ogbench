from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from envs.robomanip.lie import SE3
from envs.robomanip.traj import piecewise_polynomial, piecewise_quaternion


@dataclass
class PiecewisePose:
    position_trajectory: piecewise_polynomial.PiecewisePolynomial
    orientation_trajectory: piecewise_quaternion.PiecewiseQuaternionSlerp

    @staticmethod
    def make_linear(
        times: np.ndarray,
        poses: Sequence[SE3],
    ) -> PiecewisePose:
        pos = [p.translation() for p in poses]
        pos_traj = piecewise_polynomial.FirstOrderHold(times, pos)
        rot = [p.rotation() for p in poses]
        ori_traj = piecewise_quaternion.PiecewiseQuaternionSlerp(times, rot)
        return PiecewisePose(pos_traj, ori_traj)

    def value(self, t):
        ori = self.orientation_trajectory.value(t)
        pos = self.position_trajectory.value(t)
        return SE3.from_rotation_and_translation(ori, pos)
