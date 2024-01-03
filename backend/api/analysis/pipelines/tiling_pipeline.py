from api.analysis.pipelines.base_processing_pipeline import BaseProcessingPipeline
from api.analysis.processing.preprocessing import load_images_dataset, pad_dataset, split_images_into_patches
from api.analysis.processing.postprocessing import recombine_patches
import logging

logger = logging.getLogger(__name__)

class TilingPipeline(BaseProcessingPipeline):
    DEFAULT_BATCH_SIZE = 64
    def __init__(self, model, patch_size, batch_size=DEFAULT_BATCH_SIZE):
        if len(patch_size) != 3 or not all(isinstance(dim, int) and dim > 0 for dim in patch_size):
            raise ValueError("patch_size must be a tuple of three positive integers")
        
        if not isinstance(batch_size, int) and batch_size < 1:
            raise ValueError("batch_size must be a positive integer")
        
        super().__init__(model)
        self.patch_size = patch_size
        self.batch_size = batch_size

    def process(self, input_images):
        logger.debug(f"Starting TilingPipeline")
        logger.debug(f"Loading images...")
        images_dataset = load_images_dataset(input_images)
        
        logger.debug(f"Padding...")
        padded_dataset, original_shapes = pad_dataset(images_dataset, self.patch_size)
        
        logger.debug(f"Splitting...")
        patched_images_dataset, patch_counts = split_images_into_patches(padded_dataset, self.patch_size)
        
        logger.debug(f"Batching images into batches of size {self.batch_size}...")
        patched_images_dataset = patched_images_dataset.batch(self.batch_size)
        
        logger.debug(f"Making predictions...")
        predictions = self.model.predict(patched_images_dataset) # numpy array of shape (n, patch_size, patch_size, 1) float32 in range [0, 1]
        
        logger.debug(f"Recombining...")
        return recombine_patches(predictions, original_shapes, patch_counts, self.patch_size), images_dataset
    
    def load_model(self):
        super().load_model()