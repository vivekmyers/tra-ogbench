from __future__ import annotations

import abc


class Trajectory(abc.ABC):
    @abc.abstractmethod
    def start_time(self) -> float:
        raise NotImplementedError

    @abc.abstractmethod
    def end_time(self) -> float:
        raise NotImplementedError

    # @abc.abstractmethod
    # def rows(self) -> int:
    #     raise NotImplementedError

    # @abc.abstractmethod
    # def cols(self) -> int:
    #     raise NotImplementedError

    # @abc.abstractmethod
    # def make_derivative(self, order: int = 1) -> Trajectory:
    #     raise NotImplementedError

    # @abc.abstractmethod
    # def eval_derivative(self, t, order: int = 1):
    #     raise NotImplementedError

    # @abc.abstractmethod
    # def has_derivative(self) -> bool:
    #     raise NotImplementedError

    @abc.abstractmethod
    def value(self, t):
        raise NotImplementedError
