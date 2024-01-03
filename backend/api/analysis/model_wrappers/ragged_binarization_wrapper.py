from .abstract_model_wrapper import AbstractModelWrapper
import tensorflow as tf

class RaggedBinarizationWrapper(AbstractModelWrapper):
    def __init__(self, model, threshold=0.5):
        self.model = model
        self.threshold = threshold
        
    def load_model(self, model_path=None):
        return self.model.load_model(model_path)
    
    def predict(self, input_data):
        predictions_ragged_tensor = self.model.predict(input_data)
        tensor = predictions_ragged_tensor
        tensor = predictions_ragged_tensor.to_tensor()
        return tf.cast(tf.where(tensor > self.threshold, 255, 0), tf.uint8)