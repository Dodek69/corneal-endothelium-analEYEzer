from api.analysis.processing.preprocessing import load_image
from api.analysis.pipelines.base_processing_pipeline import BaseProcessingPipeline
import tensorflow as tf
import os

class MockModelPipeline(BaseProcessingPipeline):
    def __init__(self, prediction_image_path):
        if not os.path.exists(prediction_image_path):
            raise FileNotFoundError(f"Mock prediction image not found at {prediction_image_path}")
        self.prediction_image_path = prediction_image_path

    def process(self, image=None, **kwargs):
        return self.load_prediction_from_file()

    def load_prediction_from_file(self):
        bytes = tf.io.read_file(self.prediction_image_path)
        image = image = tf.io.decode_image(bytes, channels=1, dtype=tf.float32)
        image_2d = tf.squeeze(image, axis=-1)
        dataset = tf.data.Dataset.from_tensor_slices([image_2d]) # dataset of bytes
        return [image_2d.numpy()], dataset