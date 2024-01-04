from api.analysis.pipelines.base_processing_pipeline import BaseProcessingPipeline
from api.analysis.processing.preprocessing import load_images_dataset, resize_with_pad, get_image_dimensions
import logging
from api.analysis.processing.postprocessing import calculate_padding_and_resize

logger = logging.getLogger(__name__)

class ResizeWithPadPipeline(BaseProcessingPipeline):
    DEFAULT_BATCH_SIZE = 32
    def __init__(self, model, target_dimensions, batch_size=DEFAULT_BATCH_SIZE):
        if len(target_dimensions) != 2 or not all(isinstance(dim, int) and dim > 0 for dim in target_dimensions):
            raise ValueError("target_dimensions must be a tuple of two positive integers")
        
        if not isinstance(batch_size, int) and batch_size < 1:
            raise ValueError("batch_size must be a positive integer")
        
        super().__init__(model)
        self.target_dimensions = target_dimensions
        self.batch_size = batch_size

    def process(self, input_images):
        logger.debug(f"Starting ResizeWithPadPipeline")
        logger.debug(f"Loading images...")
        images_dataset = load_images_dataset(input_images)
        
        logger.debug(f"Resizing with padding to shape {self.target_dimensions}...")
        resized_dataset = images_dataset.map(lambda image: resize_with_pad(image, self.target_dimensions[0], self.target_dimensions[1]))
        
        logger.debug(f"Batching images into batches of size {self.batch_size}...")
        batched_resized_dataset = resized_dataset.batch(self.batch_size)
        
        logger.debug(f"Making predictions...")
        predictions = self.model.predict(batched_resized_dataset) # numpy array of shape (n, patch_size, patch_size, 1) float32 in range [0, 1]
        
        logger.debug(f"Resizing predictions to original shapes...")
        original_shapes = images_dataset.map(get_image_dimensions)
        original_shaped_predictions = []
        for original_shape, prediction in zip(original_shapes, predictions):
            original_shaped_prediction = calculate_padding_and_resize(prediction, self.target_dimensions[0], self.target_dimensions[1], original_shape[0], original_shape[1])
            original_shaped_predictions.append(original_shaped_prediction)
        
        return original_shaped_predictions, images_dataset
    
    def load_model(self):
        super().load_model()