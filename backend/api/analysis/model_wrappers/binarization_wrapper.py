from .abstract_model_wrapper import AbstractModelWrapper
import tensorflow as tf

class BinarizationWrapper(AbstractModelWrapper):
    def __init__(self, model, threshold=0.5):
        self.model = model
        self.threshold = threshold
        
    def load_model(self, model_path=None):
        return self.model.load_model(model_path)
    
    def predict(self, input_data):
        predictions = self.model.predict(input_data)
        # binarize using tensorflow so that there are 0 and 255 values of uint8 type
        return tf.cast(tf.where(predictions > self.threshold, 255, 0), tf.uint8)
        