from __future__ import annotations

import abc


class Trajectory(abc.ABC):
    @abc.abstractmethod
    def start_time(self) -> float:
        raise NotImplementedError

    @abc.abstractmethod
    def end_time(self) -> float:
        raise NotImplementedError

    @abc.abstractmethod
    def value(self, t):
        raise NotImplementedError
