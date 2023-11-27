from api.analysis.pipelines.base_processing_pipeline import BaseProcessingPipeline
from api.analysis.processing.preprocessing import load_images_dataset, pad_dataset, split_images_into_patches
from api.analysis.processing.postprocessing import recombine_patches
import logging

logger = logging.getLogger(__name__)

class PatchedToSingleChannelPipeline(BaseProcessingPipeline):
    DEFAULT_THRESHOLD = 0.1
    DEFAULT_BATCH_SIZE = 64
    def __init__(self, model, threshold=DEFAULT_THRESHOLD, batch_size=DEFAULT_BATCH_SIZE):
        if threshold < 0 or threshold > 1:
            raise ValueError("threshold must be a float between 0 and 1")
        
        super().__init__(model)
        
        self.threshold = threshold
        
        self.batch_size = batch_size or self.model.model.input_shape[0]

    def process(self, files, threshold=None, batch_size=None, **kwargs):
        threshold = threshold or self.threshold
        batch_size = batch_size or self.batch_size
        patch_size = self.model.model.input_shape[1:3]
        logger.debug(f"Loading images...")
        images_dataset = load_images_dataset(files)
        logger.debug(f"Padding...")
        padded_dataset, original_shapes = pad_dataset(images_dataset, patch_size, **kwargs)
        logger.debug(f"Splitting...")
        patched_images_dataset, patch_counts = split_images_into_patches(padded_dataset, patch_size)
        patched_images_dataset = patched_images_dataset.batch(self.batch_size)
        logger.debug(f"Making predictions...")
        predictions = self.model.predict(patched_images_dataset) # numpy array of shape (n, patch_size, patch_size, 1) float32 in range [0, 1]
        
        logger.debug(f"Recombining...")
        return recombine_patches(predictions, original_shapes, patch_counts, patch_size), images_dataset
    
    def load_model(self):
        super().load_model()