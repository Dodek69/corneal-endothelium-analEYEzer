import logging
from api.analysis.model_wrappers.tensorflow_model_wrapper import TensorFlowModelWrapper
from api.analysis.pipelines.patched_to_single_channel_pipeline import PatchedToSingleChannelPipeline
from api.analysis.pipelines.mock_model_pipeline import MockModelPipeline
import tensorflow as tf
import io
from PIL import Image
import numpy as np
from api.analysis.processing.postprocessing import calculate_metrics, close_and_skeletonize, binarize_and_convert, overlay_masks, visualize_labels
from api.utils.path_utils import generate_output_path
from api.analysis.services.abstract_service import AbstractService

from api.analysis.processing.image_utils import save_image
import cv2
logger = logging.getLogger(__name__)

pipelines_registry = {
    'fixed30': PatchedToSingleChannelPipeline(TensorFlowModelWrapper('api/analysis/models/fixed30')),
    'mock': MockModelPipeline('api/analysis/models/mock.png'),
}

class AnalysisService(AbstractService):
    def zip_encode(img, base_file_path, relative_file_path):
        result, buffer = cv2.imencode('.png', img)
        if result == False:
            raise Exception("could not encode image!")
        image_buffer = io.BytesIO(buffer)
        return (image_buffer.getvalue(), str(generate_output_path(base_file_path, relative_file_path)))
    
    @staticmethod
    def process(files, file_paths, masks, predictionsPath, overlayedPath, areaParameter, generateLabelledImages, labelledImagePath):
        try:
            logger.debug(f'Length of masks: {len(masks)}')
            logger.debug(f'Processing {len(files)} images')
            if not files:
                return None, 'No files provided'
            
            if len(files) != len(file_paths):
                return None, 'Number of files and file paths do not match'

            predictions, original_images = pipelines_registry.get('fixed30').process(files)
            logger.debug(f'Pipeline finished')
            
            processed_data = []
            metrics = ""
            
            for original_image, reference, prediction, file_path in zip(original_images, masks, predictions, file_paths):
                logger.debug(f"Overlaying mask")
                
                overlayed_image = overlay_masks(original_image, tf.io.decode_image(reference, channels=1, dtype=tf.float32) if reference else tf.zeros([original_image.shape[0], original_image.shape[1], original_image.shape[2]]), prediction)
                logger.debug(f"Overlayed mask")
                overlayed_image = tf.cast((overlayed_image * 255), tf.uint8).numpy()
                logger.debug(f"Converting prediction")
                prediction = binarize_and_convert(prediction, threshold=0.1).numpy()
                logger.debug(f"Converted prediction")
                
                prediction = close_and_skeletonize(prediction)
                processed_data.append(AnalysisService.zip_encode((prediction * 255).astype(np.uint8), file_path, predictionsPath))
                
                # Save image to bytes buffer
                print(f"Generate overlay for {file_path}")
                processed_data.append(AnalysisService.zip_encode(cv2.cvtColor(overlayed_image, cv2.COLOR_BGR2RGB), file_path, overlayedPath))
                
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
                logger.debug(f"generate labelled images: {generateLabelledImages}")
                if generateLabelledImages:
                    labels_visualization_image = visualize_labels(original_image, labelled_image, three_label_meetings)
                    print(f"Generate overlay for {file_path}")
                    processed_data.append(AnalysisService.zip_encode(labels_visualization_image, file_path, labelledImagePath))
             
            processed_data.append((metrics.encode('utf-8'), 'metrics.txt'))
            logger.debug(f'Processed {len(files)} images')
        except Exception as e:
            return None, str(e)
        return processed_data, None