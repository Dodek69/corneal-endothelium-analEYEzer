import logging
from api.analysis.model_wrappers.tensorflow_model_wrapper import TensorFlowModelWrapper
from api.analysis.pipelines.tiling_pipeline import TilingPipeline
from api.analysis.pipelines.resize_with_pad_pipeline import ResizeWithPadPipeline
from api.analysis.pipelines.variable_shape_pipeline import VariableShapePipeline
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

logger = logging.getLogger(__name__)

model_paths = {
    'custom': 'api/analysis/models/fixed30',
    'sm_unet-fixed': 'api/analysis/models/cea_model_type=sm_unet_backbone=mobilenet_num_filters=8_encoder_freeze=True_loss=dice_learning_rate=0.001_optimizer=adam_1703884072_strong_augumentations.keras',
    'sm_unet-variable': 'api/analysis/models/cea_model_type=sm_unet_backbone=mobilenet_num_filters=8_encoder_freeze=True_loss=dice_learning_rate=0.001_optimizer=adam_1704219885_strong_augumentations_input_shape=(None None 3).keras',
}

models = {
    'custom': BinarizationWrapper(TensorFlowModelWrapper(model_paths['custom']).load_model(), threshold=0.1),
    'sm_unet-fixed': BinarizationWrapper(TensorFlowModelWrapper(model_paths['sm_unet-fixed']).load_model(), threshold=0.15),
    'sm_unet-variable': BinarizationWrapper(TensorFlowModelWrapper(model_paths['sm_unet-variable']).load_model(), threshold=0.15),
    'sm-unet-variable-ragged': RaggedBinarizationWrapper(TensorFlowModelWrapper(model_paths['sm_unet-variable']).load_model(), threshold=0.15),
}

pipelines_registry = {
    'custom-resize_with_pad 128x128': ResizeWithPadPipeline(models['custom'], target_dimensions=(128, 128)),
    'custom-tiling 128x128': TilingPipeline(models['custom'], patch_size=(128, 128, 3)),
    
    'sm_unet-resize_with_pad 512x512': ResizeWithPadPipeline(models['sm_unet-fixed'], target_dimensions=(512, 512)),
    'sm_unet-tiling 512x512': TilingPipeline(models['sm_unet-fixed'], patch_size=(512, 512, 3)),

    'sm_unet-variable-resize_with_pad 1024x1024': ResizeWithPadPipeline(models['sm_unet-variable'], target_dimensions=(1024, 1024)),
    'sm_unet-variable-resize_with_pad 512x512': ResizeWithPadPipeline(models['sm_unet-variable'], target_dimensions=(512, 512)),
    'sm_unet-variable-resize_with_pad 128x128': ResizeWithPadPipeline(models['sm_unet-variable'], target_dimensions=(128, 128)),
    'sm_unet-variable-dynamic_resize_with_pad': DynamicResizeWithPadPipeline(models['sm_unet-variable'], downsampling_factor=64),
    'sm_unet-variable-variable': VariableShapePipeline(models['sm-unet-variable-ragged'], downsampling_factor=64),
    'sm_unet-variable-tiling 128x128': TilingPipeline(models['sm_unet-variable'], patch_size=(128, 128, 3)),
    'sm_unet-variable-tiling 512x512': TilingPipeline(models['sm_unet-variable'], patch_size=(512, 512, 3)),
    'sm_unet-variable-tiling 1024x1024': TilingPipeline(models['sm_unet-variable'], patch_size=(1024, 1024, 3)),
}

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
    
def pipeline_factory(pipeline_type, model_wrapper, patch_size=None, target_dimensions=None, downsampling_factor=None):
    if pipeline_type == "tiling":
        return TilingPipeline(model_wrapper, patch_size=patch_size)
    elif pipeline_type == "resize_with_pad":
        return ResizeWithPadPipeline(model_wrapper, target_dimensions=target_dimensions)
    elif pipeline_type == "dynamic_resize_with_pad":
        return DynamicResizeWithPadPipeline(model_wrapper, downsampling_factor=downsampling_factor)
    elif pipeline_type == "variable":
        return VariableShapePipeline(model_wrapper, downsampling_factor=downsampling_factor)
    else:
        raise ValueError("Unsupported pipeline type")

class AnalysisService(AbstractService):
    def zip_encode(img, base_file_path, relative_file_path):
        result, buffer = cv2.imencode('.png', img)
        if result == False:
            raise Exception("could not encode image!")
        
        return (base64.b64encode(buffer).decode('utf-8'), str(generate_output_path(base_file_path, relative_file_path)))
    
    @staticmethod
    def process(task_id, input_images, input_images_paths, input_masks, predictions_path, overlayed_path, area_per_pixel, generate_labelled_images, labelled_images_path, model, model_file_extension, pipeline_type, target_dimensions, downsampling_factor):
        try:
            logger.debug(f'task_id: {task_id}')
            logger.debug(f'AnalysisService started for {len(input_images)} images')
            
            if model not in pipelines_registry:
                logger.debug(f'Loading custom model from object: {model}')
                
                with tempfile.NamedTemporaryFile(suffix=model_file_extension) as tmp_file:
                    logger.debug(f"Downloading model to {tmp_file.name}...")
                    minio_repo.download_file(model, tmp_file.name)
                    
                    if model_file_extension == 'zip':
                        with tempfile.TemporaryDirectory() as tmp_dir:
                            logger.debug(f"Unzipping model to {tmp_dir}...")
                            
                            with zipfile.ZipFile(tmp_file.name, 'r') as zip_ref:
                                zip_ref.extractall(tmp_dir)
                            """
                            logger.debug(f"listing files...")
                            
                            for root, dirs, filenames in os.walk(tmp_dir):
                                logger.debug(f"Directory: {root}")
                                for file in filenames:
                                    logger.debug(f" - File: {file}")
                            """        
                            logger.debug(f"Creating pipeline...")
                            pipeline = pipeline_factory(pipeline_type, model_wrapper_factory(tmp_dir, 'tensorflow'), target_dimensions=target_dimensions, downsampling_factor=downsampling_factor)
                    else:
                        logger.debug(f"Creating pipeline...")
                        pipeline = pipeline_factory(pipeline_type, model_wrapper_factory(tmp_file.name, 'tensorflow'), target_dimensions=target_dimensions, downsampling_factor=downsampling_factor) 
                    logger.info(f"Loaded model input shape: {pipeline.model.model.input_shape}")
                    logger.info(f"Loaded model output shape: {pipeline.model.model.output_shape}")
                 
            else:
                logger.debug(f'Loading model from registry...')
                pipeline = pipelines_registry.get(model)
                
            logger.debug('Starting pipeline...')
            predictions, original_images = pipeline.process(input_images)
            logger.debug('Pipeline finished')
            
            processed_data = []
            metrics = ""
            
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
                metrics = metrics + f"{file_path}\n"
                
                logger.debug(f"Calculating metrics for {file_path}")
                skeleton_inverted = np.invert(prediction).astype(np.uint8)
                cell_density, coefficient_value, hexagonal_cell_ratio, feature_counts, three_label_meetings, num_labels, labelled_image = calculate_metrics(skeleton_inverted, area_per_pixel)

                logger.debug(f"Writing metrics...")
                metrics = metrics + f"Cell Density: {cell_density}\n"
                metrics = metrics + f"Coefficient Value: {coefficient_value}\n"
                metrics = metrics + f"Hexagonal Cell Ratio: {hexagonal_cell_ratio}\n"
                metrics = metrics + f"====\n"
                
                if generate_labelled_images:
                    logger.debug(f"Generating labelled image...")
                    labels_visualization_image = visualize_labels(original_image, labelled_image, three_label_meetings)
                    
                    logger.debug(f"Encoding labelled image...")
                    processed_data.append(AnalysisService.zip_encode(labels_visualization_image, file_path, labelled_images_path))
             
            #processed_data.append((metrics.encode('utf-8'), 'metrics.txt'))
            logger.debug(f'Metrics: {metrics}')
            
            for index, (data, filename) in enumerate(processed_data):
                ordered_filename = f"{task_id}/{index:03d}_{filename}"
                image_data_bytes = base64.b64decode(data.encode('utf-8'))
                minio_repo.upload_file_directly(image_data_bytes, ordered_filename)
            logger.debug(f'Finished AnalysisService for {len(input_images)} images')
            
        except Exception as e:
            return None, str(e)
        return processed_data, None