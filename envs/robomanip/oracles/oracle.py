import abc


class Oracle(abc.ABC):
    @abc.abstractmethod
    def reset(self, obs, info): ...

    @abc.abstractmethod
    def select_action(self, obs, info): ...

    @property
    @abc.abstractmethod
    def done(self) -> bool: ...
