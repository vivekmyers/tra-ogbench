from __future__ import annotations

from scipy.interpolate import interp1d

from envs.robomanip.traj.piecewise_trajectory import PiecewiseTrajectory


class PiecewisePolynomial(PiecewiseTrajectory):
    """Represents a list of contiguous segments in time with polynomials defined at
    each segment.
    """


class ZeroOrderHold(PiecewisePolynomial):
    """A piecewise constant PiecewisePolynomial."""

    def value(self, t):
        segment_index = self.get_segment_index(t)
        return self.samples[segment_index]


class FirstOrderHold(PiecewisePolynomial):
    """A piecewise linear PiecewisePolynomial."""

    def __init__(self, breaks, samples):
        super().__init__(breaks, samples)

        self._interp = interp1d(
            self.breaks, self.samples, kind="linear", axis=0, assume_sorted=True
        )

    def value(self, t):
        return self._interp(t)
