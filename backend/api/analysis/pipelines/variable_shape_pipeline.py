from api.analysis.pipelines.base_processing_pipeline import BaseProcessingPipeline
from api.analysis.processing.preprocessing import load_images_dataset, add_dimension, resize_to_next_divisor, get_image_dimensions
import logging
from api.analysis.processing.postprocessing import calculate_padding_and_resize

logger = logging.getLogger(__name__)

class VariableShapePipeline(BaseProcessingPipeline):
    DEFAULT_BATCH_SIZE = 64
    def __init__(self, model, downsampling_factor, batch_size=DEFAULT_BATCH_SIZE):
        if downsampling_factor < 1 or not isinstance(downsampling_factor, int):
            raise ValueError("downsampling_factor must be a positive integer")
            
        if not isinstance(batch_size, int) and batch_size < 1:
            raise ValueError("batch_size must be a positive integer")
        
        super().__init__(model)
        self.downsampling_factor = downsampling_factor
        self.batch_size = batch_size

    def process(self, input_images):
        logger.debug(f"Starting VariableShapePipeline")
        logger.debug(f"Loading images...")
        images_dataset = load_images_dataset(input_images)
        
        logger.debug(f"Resizing with padding...")
        resized_images_and_original_shapes = images_dataset.map(lambda image: resize_to_next_divisor(image, divisor=self.downsampling_factor))
        resized_dataset = resized_images_and_original_shapes.map(lambda x, y: x)
        original_shapes = resized_images_and_original_shapes.map(lambda x, y: y)
        
        resized_shapes = resized_dataset.map(get_image_dimensions)
        for resized_shape, original_shape in zip(resized_shapes, original_shapes):
            logger.debug(f"resized_shape: {resized_shape}, original_shape: {original_shape}")

        logger.debug(f"Adding dimension...")
        resized_dataset = resized_dataset.batch(1)
        
        logger.debug(f"Making predictions...")
        predictions = self.model.predict(resized_dataset) # numpy array of shape (n, patch_size, patch_size, 1) float32 in range [0, 1]
        
        logger.debug(f"Resizing predictions...")
        original_shaped_predictions = []
        for original_shape, resized_shape, prediction in zip(original_shapes, resized_shapes, predictions):
            original_shaped_prediction = calculate_padding_and_resize(prediction, resized_shape[0], resized_shape[1], original_shape[0], original_shape[1])
            original_shaped_predictions.append(original_shaped_prediction)
        
        return original_shaped_predictions, images_dataset
    
    def load_model(self):
        super().load_model()