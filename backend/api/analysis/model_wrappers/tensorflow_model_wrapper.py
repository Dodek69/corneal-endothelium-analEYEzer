from .abstract_model_wrapper import AbstractModelWrapper
import tensorflow as tf
import logging

logger = logging.getLogger(__name__)

class TensorFlowModelWrapper(AbstractModelWrapper):
    def __init__(self, model_path):
        self.load_model(model_path)
        
    def load_model(self, model_path):
        logger.info(f"Loading model from {model_path}")
        self.model = tf.keras.models.load_model(model_path)
        logger.info(f"Loaded model input shape: {self.model.input_shape}")
        logger.info(f"Loaded model output shape: {self.model.output_shape}")

    def predict(self, input_data):
        return self.model.predict(input_data)