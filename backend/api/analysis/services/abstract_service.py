from abc import ABC, abstractmethod

class AbstractService(ABC):
    @abstractmethod
    def process(self):
        pass