import tensorflow as tf
from PIL import Image
import numpy as np
import logging
import api.utils.time_utils as time_utils

logger = logging.getLogger(__name__)

def save_image(image_array):
    try:
        logger.debug(f"Debug: Initial type: {type(image_array)}")
        logger.debug(f"Debug: Initial shape: {image_array.shape}")
        logger.debug(f"Debug: Initial datatype: {image_array.dtype}")

        if isinstance(image_array, tf.Tensor):
            image_array = image_array.numpy()
            
        normalized = image_array.max() <= 1

        if normalized:
            image_array = image_array * 255
            logger.debug("Debug: Expanded image values to [0, 255] range")
        
        image_array = (image_array).astype(np.uint8)
        
        if image_array.ndim == 3 and image_array.shape[-1] == 1:
            logger.debug("Debug: Reshaping")
            image_array = image_array.reshape(image_array.shape[0], image_array.shape[1])

        image = Image.fromarray(image_array)
        
        file_path = f"api/analysis/debug/{time_utils.get_timestamp_string()}.png"
        image.save(file_path)
    except Exception as e:
        logger.error(f"Error while saving image: {e}")