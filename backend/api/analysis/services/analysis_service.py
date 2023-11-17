import logging
from api.analysis.model_wrappers.tensorflow_model_wrapper import TensorFlowModelWrapper
from api.analysis.pipelines.patched_to_single_channel_pipeline import PatchedToSingleChannelPipeline
from api.analysis.pipelines.mock_model_pipeline import MockModelPipeline

import io
from PIL import Image

logger = logging.getLogger(__name__)

pipelines_registry = {
    'fixed30': PatchedToSingleChannelPipeline(TensorFlowModelWrapper('api/analysis/models/fixed30')),
    'mock': MockModelPipeline('api/analysis/models/mock.png'),
}

class AnalysisService:
    @staticmethod
    def process_images(files, file_paths):
        try:
            logger.debug(f'Processing {len(files)} images')
            if not files:
                return None, 'No files provided'
            
            if len(files) != len(file_paths):
                return None, 'Number of files and file paths do not match'

            mask_predictions, overlayed_images = pipelines_registry.get('fixed30').process(files)
            processed_data = []
            
            for prediction, overlayed_image, file_path in zip(mask_predictions, overlayed_images, file_paths):
                if prediction.shape[-1] == 1:
                    prediction = prediction[..., 0]
                
                # Convert NumPy array to image
                prediction_pil = Image.fromarray(prediction)
                
                # Save image to bytpres buffer
                image_buffer = io.BytesIO()
                prediction_pil.save(image_buffer, format='PNG')
                processed_data.append((image_buffer.getvalue(), f"predictions/{file_path}"))

                if overlayed_image.shape[-1] == 1:
                    overlayed_image = overlayed_image[..., 0]
                
                # Convert NumPy array to image
                overlayed_image_pil = Image.fromarray(overlayed_image)
                
                # Save image to bytes buffer
                image_buffer = io.BytesIO()
                overlayed_image_pil.save(image_buffer, format='PNG')
                
                processed_data.append((image_buffer.getvalue(), f"overlays/{file_path}"))
                
            logger.debug(f'Processed {len(files)} images')
        except Exception as e:
            return None, str(e)
        return processed_data, None