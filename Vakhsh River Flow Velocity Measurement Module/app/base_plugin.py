from abc import ABC, abstractmethod


class BasePlugin(ABC):
    """Base interface implemented by all UI plugins."""

    def order(self):
        return 999

    @abstractmethod
    def name(self):
        raise NotImplementedError

    @abstractmethod
    def widget(self):
        raise NotImplementedError

