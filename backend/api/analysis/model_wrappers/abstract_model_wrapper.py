from abc import ABC, abstractmethod

class AbstractModelWrapper(ABC):
    @abstractmethod
    def __init__(self, model_path):
        pass
    
    @abstractmethod
    def load_model(self):
        pass

    @abstractmethod
    def predict(self, input_data):
        pass