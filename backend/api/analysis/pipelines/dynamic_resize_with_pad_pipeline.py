from api.analysis.pipelines.base_processing_pipeline import BaseProcessingPipeline
from api.analysis.processing.preprocessing import load_images_dataset, resize_with_pad, get_image_dimensions, round_up
import logging
from api.analysis.processing.postprocessing import calculate_padding_and_resize
import tensorflow as tf
logger = logging.getLogger(__name__)

class DynamicResizeWithPadPipeline(BaseProcessingPipeline):
    DEFAULT_BATCH_SIZE = 32
    def __init__(self, model, downsampling_factor, batch_size=DEFAULT_BATCH_SIZE):
        if downsampling_factor < 1 or not isinstance(downsampling_factor, int):
            raise ValueError("downsampling_factor must be a positive integer")
            
        if not isinstance(batch_size, int) and batch_size < 1:
            raise ValueError("batch_size must be a positive integer")
        
        super().__init__(model)
        self.downsampling_factor = downsampling_factor
        self.batch_size = batch_size

    def process(self, input_images):
        logger.debug(f"Starting DynamicResizeWithPadPipeline")
        logger.debug(f"Loading images...")
        images_dataset = load_images_dataset(input_images)

        original_shapes = list(images_dataset.map(get_image_dimensions).as_numpy_iterator())

        # Get the maximum height and width
        max_height = tf.reduce_max([shape[0] for shape in original_shapes])
        max_width = tf.reduce_max([shape[1] for shape in original_shapes])

        logger.debug(f"max_height: {max_height}, max_width: {max_width}")

        target_height = round_up(max_height, self.downsampling_factor)
        target_width = round_up(max_width, self.downsampling_factor)

        logger.debug(f"Resizing with padding to shape ({target_height}, {target_width})...")
        resized_dataset = images_dataset.map(lambda image: resize_with_pad(image, target_height, target_width))

        logger.debug(f"Batching images into batches of size {self.batch_size}...")
        batched_resized_dataset = resized_dataset.batch(self.batch_size)

        logger.debug(f"Making predictions...")
        predictions = self.model.predict(batched_resized_dataset) # numpy array of shape (n, patch_size, patch_size, 1) float32 in range [0, 1]

        logger.debug(f"Resizing predictions to original shapes...")
        original_shaped_predictions = []
        for original_shape, prediction in zip(original_shapes, predictions):
            original_shaped_prediction = calculate_padding_and_resize(prediction, target_height, target_width, original_shape[0], original_shape[1])
            original_shaped_predictions.append(original_shaped_prediction)

        return original_shaped_predictions, images_dataset
    
    def load_model(self):
        super().load_model()