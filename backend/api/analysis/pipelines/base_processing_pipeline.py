from abc import ABC, abstractmethod
from api.analysis.model_wrappers.abstract_model_wrapper import AbstractModelWrapper

class BaseProcessingPipeline(ABC):
    def __init__(self, model: AbstractModelWrapper):
        if not isinstance(model, AbstractModelWrapper):
            raise TypeError(f"model must be an instance of {AbstractModelWrapper.__name__}")
        self.model = model
        
    def load_model(self):
        return self.model.load_model()
    
    @abstractmethod
    def process(self, images, **kwargs):
        pass