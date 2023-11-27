import logging
from api.analysis.model_wrappers.tensorflow_model_wrapper import TensorFlowModelWrapper
from api.analysis.pipelines.patched_to_single_channel_pipeline import PatchedToSingleChannelPipeline
from api.analysis.pipelines.mock_model_pipeline import MockModelPipeline
import tensorflow as tf
import io
from PIL import Image
import numpy as np
from api.analysis.processing.postprocessing import calculate_metrics, close_and_skeletonize
from api.analysis.processing.postprocessing import binarize_and_convert, overlay_masks
from api.utils.path_utils import generate_output_path
from api.analysis.services.abstract_service import AbstractService

logger = logging.getLogger(__name__)

pipelines_registry = {
    'fixed30': PatchedToSingleChannelPipeline(TensorFlowModelWrapper('api/analysis/models/fixed30')),
    'mock': MockModelPipeline('api/analysis/models/mock.png'),
}

class AnalysisService(AbstractService):
    @staticmethod
    def process(files, file_paths, masks, predictionsPath, overlayedPath, areaParameter):
        try:
            logger.debug(f'Length of masks: {len(masks)}')
            logger.debug(f'Processing {len(files)} images')
            if not files:
                return None, 'No files provided'
            
            if len(files) != len(file_paths):
                return None, 'Number of files and file paths do not match'

            predictions, original_images = pipelines_registry.get('fixed30').process(files)
            logger.debug(f'Pipeline finished')
            overlayed_images = []
            predictions2 = []
            for original_image, reference, prediction in zip(original_images, masks, predictions):
                logger.debug(f"Overlaying mask")
                
                overlayed_image = overlay_masks(original_image, tf.io.decode_image(reference, channels=1, dtype=tf.float32) if reference else tf.zeros([original_image.shape[0], original_image.shape[1], original_image.shape[2]]), prediction)
                logger.debug(f"Overlayed mask")
                overlayed_images.append(tf.cast(overlayed_image * 255, tf.uint8).numpy())
                logger.debug(f"Converting prediction")
                temp = binarize_and_convert(prediction, threshold=0.1).numpy()
                logger.debug(f"Converted prediction")
                predictions2.append(temp)
            
            processed_data = []
            metrics = ""
            
            for prediction, overlayed_image, file_path in zip(predictions2, overlayed_images, file_paths):
                
                if prediction.shape[-1] == 1:
                    prediction = prediction[..., 0]
                    
                prediction = close_and_skeletonize(prediction)
                
                # Convert NumPy array to image
                prediction_pil = Image.fromarray(prediction)
                
                # Save image to bytpres buffer
                image_buffer = io.BytesIO()
                prediction_pil.save(image_buffer, format='PNG')
                processed_data.append((image_buffer.getvalue(), str(generate_output_path(file_path, predictionsPath))))
                    
                if overlayed_image.shape[-1] == 1:
                    overlayed_image = overlayed_image[..., 0]
                
                # Convert NumPy array to image
                overlayed_image_pil = Image.fromarray(overlayed_image)
                
                # Save image to bytes buffer
                image_buffer = io.BytesIO()
                overlayed_image_pil.save(image_buffer, format='PNG')
                print(f"Generate overlay for {file_path}")
                processed_data.append((image_buffer.getvalue(), str(generate_output_path(file_path, overlayedPath))))
                print(f"Generated overlay path {str(generate_output_path(file_path, overlayedPath))}")
                metrics = metrics + f"{file_path}\n"
                logger.debug(f"Calculating metrics for {file_path}")
                skeleton_inverted = np.invert(prediction).astype(np.uint8)
                cell_density, coefficient_value, hexagonal_cell_ratio, feature_counts, three_label_meetings, num_labels, labelled_image = calculate_metrics(skeleton_inverted, areaParameter)

                logger.debug(f"Calculated metrics for {file_path}")
                metrics = metrics + f"Cell Density: {cell_density}\n"
                metrics = metrics + f"Coefficient Value: {coefficient_value}\n"
                metrics = metrics + f"Hexagonal Cell Ratio: {hexagonal_cell_ratio}\n"
                metrics = metrics + f"====\n"
            
            processed_data.append((metrics.encode('utf-8'), 'metrics.txt'))
            logger.debug(f'Processed {len(files)} images')
        except Exception as e:
            return None, str(e)
        return processed_data, None