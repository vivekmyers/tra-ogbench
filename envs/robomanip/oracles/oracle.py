import abc


class Oracle(abc.ABC):
    @abc.abstractmethod
    def reset(self, obs): ...

    @abc.abstractmethod
    def select_action(self, obs): ...

    @property
    @abc.abstractmethod
    def done(self) -> bool: ...
