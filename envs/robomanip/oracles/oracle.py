import abc


class Oracle(abc.ABC):
    @abc.abstractmethod
    def reset(self, ob, info): ...

    @abc.abstractmethod
    def select_action(self, ob, info): ...

    @property
    @abc.abstractmethod
    def done(self) -> bool: ...
