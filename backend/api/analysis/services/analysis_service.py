import logging
from api.analysis.model_wrappers.tensorflow_model_wrapper import TensorFlowModelWrapper
from api.analysis.pipelines.tiling_pipeline import TilingPipeline
from api.analysis.pipelines.resize_with_pad_pipeline import ResizeWithPadPipeline
from api.analysis.pipelines.dynamic_resize_with_pad_pipeline import DynamicResizeWithPadPipeline
import tensorflow as tf
import numpy as np
from api.analysis.processing.postprocessing import calculate_metrics, close_and_skeletonize, overlay_masks, visualize_labels
from api.utils.path_utils import generate_output_path
from api.analysis.services.abstract_service import AbstractService
import cv2
import base64
from api.analysis.repositories.minio_repository import MinioRepository
import zipfile
import tempfile
from api.analysis.model_wrappers.binarization_wrapper import BinarizationWrapper
from api.analysis.model_wrappers.ragged_binarization_wrapper import RaggedBinarizationWrapper
from api.analysis.registers import pipelines_registry, available_pipelines
import os
import glob

logger = logging.getLogger(__name__)


minio_repo = MinioRepository(
    endpoint_url='http://minio:9000',
    access_key='minio',
    secret_key='minio123',
    bucket_name='corneal-endothelium-analeyezer'
)

def model_wrapper_factory(model_path, model_type):
    if model_type == "tensorflow":
        return TensorFlowModelWrapper(model_path).load_model()
    else:
        raise ValueError("Unsupported model type")
    
def pipeline_factory(pipeline_type, model_wrapper, target_dimensions=None, downsampling_factor=None):
    pipeline = available_pipelines[pipeline_type]
    if pipeline == TilingPipeline:
        logger.debug(f"Creating tiling pipeline with patch_size: {target_dimensions}")
        return pipeline(model_wrapper, patch_size=target_dimensions + [3])
    elif pipeline == ResizeWithPadPipeline:
        logger.debug(f"Creating resize with pad pipeline with target_dimensions: {target_dimensions}")
        return pipeline(model_wrapper, target_dimensions=target_dimensions)
    elif pipeline == DynamicResizeWithPadPipeline:
        logger.debug(f"Creating dynamic resize with pad pipeline with downsampling_factor: {downsampling_factor}")
        return pipeline(model_wrapper, downsampling_factor=downsampling_factor)
    else:
        raise ValueError("Unsupported pipeline type")
    

class AnalysisService(AbstractService):
    def zip_encode(img, base_file_path, relative_file_path):
        result, buffer = cv2.imencode('.png', img)
        if result == False:
            raise Exception("could not encode image!")
        
        return (base64.b64encode(buffer).decode('utf-8'), str(generate_output_path(base_file_path, relative_file_path)))
    
    @staticmethod
    def process(task_id, input_images, input_images_paths, input_masks, predictions_path, overlayed_path, area_per_pixel, generate_labelled_images, labelled_images_path, pipeline_name, custom_model_object_name, custom_model_extension, pipeline_type, target_dimensions, downsampling_factor, threshold):
        logger.debug(f'AnalysisService started images with task_id {task_id}')
        
        if pipeline_name:
            logger.debug(f'Loading model from registry...')
            pipeline = pipelines_registry.get(pipeline_name)
        else:
            logger.debug(f'Custom model object name: {custom_model_object_name}')
            logger.debug(f'Custom model extension: {custom_model_extension}')
            
            with tempfile.NamedTemporaryFile(suffix=custom_model_extension) as tmp_file:
                logger.debug(f"Downloading model to {tmp_file.name}...")
                minio_repo.download_file(custom_model_object_name, tmp_file.name)
                
                if custom_model_extension == '.zip':
                    with tempfile.TemporaryDirectory() as tmp_dir:
                        
                        logger.debug(f"Unzipping model to {tmp_dir}")
                        with zipfile.ZipFile(tmp_file.name, 'r') as zip_ref:
                            zip_ref.extractall(tmp_dir)

                        logger.debug(f"listing files...")

                        # Use glob to find saved_model.pb
                        model_files = glob.glob(f"{tmp_dir}/**/saved_model.pb", recursive=True)

                        if model_files:
                            # Assuming the first match is the desired one
                            model_dir = os.path.dirname(model_files[0])
                        else:
                            # Handle the case where saved_model.pb is not found
                            logger.error("saved_model.pb not found in the unzipped files")
                            return (None, None), "saved_model.pb not found in the zip file"

                        logger.debug(f"Creating {pipeline_type}")
                        model = model_wrapper_factory(model_dir, 'tensorflow')
                else:
                    logger.debug(f"Creating {pipeline_type}")
                    model = model_wrapper_factory(tmp_file.name, 'tensorflow')
                binarized_model = BinarizationWrapper(model, threshold=threshold)
                pipeline = pipeline_factory(pipeline_type, binarized_model, target_dimensions=target_dimensions, downsampling_factor=downsampling_factor)
                logger.info(f"Loaded model input shape: {pipeline.model.model.model.input_shape}")
                logger.info(f"Loaded model output shape: {pipeline.model.model.model.output_shape}")
            
        logger.debug('Starting pipeline...')
        predictions, original_images = pipeline.process(input_images)
        logger.debug('Pipeline finished')
        
        processed_data = []
        metrics = []
        
        for original_image, reference, prediction, file_path in zip(original_images, input_masks, predictions, input_images_paths):
            logger.debug(f"Overlaying mask...")
            overlayed_image = overlay_masks(original_image, tf.io.decode_image(reference, channels=1, dtype=tf.float32) if reference else tf.zeros([original_image.shape[0], original_image.shape[1], original_image.shape[2]]), prediction)
            
            logger.debug(f"Casting overlayed image...")
            overlayed_image = tf.cast((overlayed_image * 255), tf.uint8).numpy()
            prediction = prediction.numpy()
            
            logger.debug(f"Closing and skeletonizing prediction...")
            prediction = close_and_skeletonize(prediction)
            
            logger.debug(f"Encoding prediction...")
            processed_data.append(AnalysisService.zip_encode((prediction * 255).astype(np.uint8), file_path, predictions_path))
            
            logger.debug(f"Encoding overlayed image...")
            processed_data.append(AnalysisService.zip_encode(cv2.cvtColor(overlayed_image, cv2.COLOR_BGR2RGB), file_path, overlayed_path))
            
            logger.debug(f"Generated overlay path {str(generate_output_path(file_path, overlayed_path))}")
            
            logger.debug(f"Calculating metrics for {file_path}")
            skeleton_inverted = np.invert(prediction).astype(np.uint8)
            num_labels, area, cell_density, std_areas, mean_areas, coefficient_value, num_hexagonal, hexagonal_cell_ratio, feature_counts, three_label_meetings, num_labels, labelled_image = calculate_metrics(skeleton_inverted, area_per_pixel)

            logger.debug(f"Appending metrics...")
            metrics.append((num_labels, area, cell_density, std_areas, mean_areas, coefficient_value, num_hexagonal, hexagonal_cell_ratio))
            
            if generate_labelled_images:
                logger.debug(f"Generating labelled image...")
                labels_visualization_image = visualize_labels(original_image, labelled_image, three_label_meetings)
                
                logger.debug(f"Encoding labelled image...")
                processed_data.append(AnalysisService.zip_encode(labels_visualization_image, file_path, labelled_images_path))
            
        logger.debug(f'Metrics: {metrics}')
        
        max_retries = 5
        for retry in range(max_retries):
            for index, (data, filename) in enumerate(processed_data):
                #logger.debug(f'generating ordered filename...')
                ordered_filename = f"{task_id}\{index:03d}\{filename}"
                
                #logger.debug(f'decoding data...')
                image_data_bytes = base64.b64decode(data.encode('utf-8'))
                
                #logger.debug(f'uploading file...')
                minio_repo.upload_file_directly(image_data_bytes, ordered_filename)

            logger.debug(f'listing files...')
            files = minio_repo.list_files()
            task_files = [f for f in files if f.startswith(f"{task_id}\\")]

            if len(task_files) == len(processed_data):
                logger.debug(f"upload successful after {retry} retries")
                return (None, metrics), None
            else:
                logger.debug(f"retrying upload after {retry} retries")

        logger.debug(f"upload failed after {max_retries} retries, returning processed data via message broker")
        return (processed_data, metrics), None