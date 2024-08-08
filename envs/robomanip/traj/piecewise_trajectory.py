import abc

import numpy as np

from envs.robomanip.traj import trajectory


class PiecewiseTrajectory(trajectory.Trajectory, abc.ABC):
    """Abstract class that implements the basic logic of maintaining consequent
    segments of time (delimited by breaks) to implement a trajectory that is
    represented by simpler logic in each segment or piece."""

    def __init__(self, breaks, samples):
        self.breaks = np.asarray(breaks)
        self.samples = np.asarray(samples)

        assert len(self.breaks) >= 2
        assert len(self.breaks) == len(self.samples)
        assert np.diff(self.breaks).all() > 0

        self._t_min = self.breaks[0]

    def get_segment_index(self, t):
        return np.where(
            t >= self.breaks[-1],
            self.n_segments - 1,
            np.searchsorted(self.breaks, t, side='right') - 1,
        )

    @property
    def n_segments(self) -> int:
        return len(self.breaks) - 1

    def start_time(self, segment_index: int | None = None) -> float:
        if segment_index is not None:
            assert 0 <= segment_index <= self.n_segments - 1
            return self.breaks[segment_index]
        return self.breaks[0]

    def end_time(self, segment_index: int | None = None) -> float:
        if segment_index is not None:
            assert 0 <= segment_index <= self.n_segments - 1
            return self.breaks[segment_index + 1]
        return self.breaks[self.n_segments - 1]

    def duration(self, segment_index: int | None = None) -> float:
        return self.end_time(segment_index) - self.start_time(segment_index)
