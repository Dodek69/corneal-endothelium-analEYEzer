from api.analysis.processing.image_utils import save_image
import numpy as np
import tensorflow as tf
import logging

logger = logging.getLogger(__name__)

def binarize_and_convert(arr, threshold=0.5):
    # Binarize based on the threshold
    binarized = tf.where(arr > threshold, 255, 0)

    # convert to uint8
    binarized = tf.cast(binarized, tf.uint8)
    
    return binarized.numpy()

def overlay_mask_to_pil_image(image, mask, color=(1, 0, 0), alpha=0.01):
    # Convert the image tensor to a numpy array if it's not already
    image = image.numpy()

    # Create an RGB version of the mask
    colored_mask = np.zeros_like(image)
    colored_mask[:, :, 0] = mask * color[0]  # Red channel
    colored_mask[:, :, 1] = mask * color[1]  # Green channel
    colored_mask[:, :, 2] = mask * color[2]  # Blue channel

    # Overlay the mask on the image
    overlayed_image = np.clip(image * (1 - alpha) + colored_mask * alpha, 0, 1)

    overlayed_image = (overlayed_image * 255).astype(np.uint8)

    return overlayed_image


def recombine_patches(predictions, images_dataset, original_shapes, patch_counts, patch_size, threshold=0.5):
    recombined_images = []
    start_index = 0
    
    for (original_height, original_width), patch_count in zip(original_shapes, patch_counts):
        
        # Calculate the number of patches along each dimension
        num_patches_height = int(np.ceil(original_height / patch_size[1]))
        num_patches_width = int(np.ceil(original_width / patch_size[0]))
        logger.debug(f"calculated patch_size: {(num_patches_height * num_patches_width)}")
        current_patches = predictions[start_index:start_index + num_patches_height * num_patches_width]
        logger.debug(f"current_patches.shape: {current_patches.shape}")
        start_index += patch_count
        
        # Reshape the patches to align in a grid
        reshaped_patches = tf.reshape(current_patches, [num_patches_height, num_patches_width, patch_size[0], patch_size[1], 1])

        # Rearrange the patches and merge them
        recombined_image = tf.transpose(reshaped_patches, [0, 2, 1, 3, 4])
        recombined_image = tf.reshape(recombined_image, [num_patches_height * patch_size[0], num_patches_width * patch_size[1]])

        # crop image to remove padding
        recombined_image = recombined_image[:original_height, :original_width]
        
        # Add to the list of recombined images
        recombined_images.append(recombined_image)
        
    predictions = []
    overlayed_images = []
    
    for image, recombined in zip(images_dataset, recombined_images):
        prediction = binarize_and_convert(recombined, threshold=threshold)
        predictions.append(prediction)
        overlayed_image = overlay_mask_to_pil_image(image, prediction)
        overlayed_images.append(overlayed_image)
    return predictions, overlayed_images