import numpy as np
import tensorflow as tf
import logging
import cv2
from skimage.morphology import skeletonize
from api.analysis.processing.image_utils import save_image

logger = logging.getLogger(__name__)
KERNEL_SIZE = (3,3)

def binarize_and_convert(arr, threshold=0.5):
    # Binarize based on the threshold
    binarized = tf.where(arr > threshold, 255, 0)

    # convert to uint8
    binarized = tf.cast(binarized, tf.uint8)
    
    return binarized

def overlay_mask_to_pil_image(image, mask, color=(1, 0, 0), alpha=0.01):
    # Convert the image tensor to a numpy array if it's not already
    image = image.numpy()
    
    logger.debug(f'Checking Type of image: {type(image)}')
    logger.debug(f'Checking Shape of image: {image.shape}')
    logger.debug(f'Checking dtype of image: {image.dtype}')
    
    logger.debug(f'Checking Type of mask: {type(mask)}')
    logger.debug(f'Checking Shape of mask: {mask.shape}')
    logger.debug(f'Checking dtype of mask: {mask.dtype}')

    # Create an RGB version of the mask
    colored_mask = np.zeros_like(image)
    colored_mask[:, :, 0] = mask * color[0]  # Red channel
    colored_mask[:, :, 1] = mask * color[1]  # Green channel
    colored_mask[:, :, 2] = mask * color[2]  # Blue channel

    # Overlay the mask on the image
    overlayed_image = np.clip(image * (1 - alpha) + colored_mask * alpha, 0, 1)

    overlayed_image = (overlayed_image * 255).astype(np.uint8)

    return overlayed_image

def overlay_masks(image, reference, prediction, reference_color = (1, 0, 0), prediction_color = (1, 1, 0), overlap_color = (0, 1, 0), alpha=0.6):
    logger.debug(f"Original image shape: {image.shape}")
    logger.debug(f"Reference shape: {reference.shape}")
    logger.debug(f"Prediction shape: {prediction.shape}")
    
    logger.debug(f"Before expand dims")
    #reference = tf.expand_dims(reference, -1)
    prediction = tf.expand_dims(prediction, -1)
    
    logger.debug(f" After expand dims")
    # Convert float masks to boolean
    mask1_bool = reference > 0.5
    mask2_bool = prediction > 0.5

    logger.debug(f" After bool")
    # Reshape masks to 3D (x, y, 1)
    mask1_3d = tf.cast(mask1_bool, tf.float32)
    mask2_3d = tf.cast(mask2_bool, tf.float32)

    logger.debug(f" After 3d")
    # Apply colors to masks
    mask1_colored = mask1_3d * reference_color
    mask2_colored = mask2_3d * prediction_color
    logger.debug(f" After colored")
    # Determine overlapping area and apply overlap color
    overlap_mask = tf.math.logical_and(mask1_bool, mask2_bool)    

    #overlap_colored = tf.cast(overlap_mask, tf.float32) * overlap_color
    logger.debug(f" After overlap")
    # Combine masks
    combined_masks = mask1_colored + mask2_colored
    logger.debug(f" After combine")
    # Apply overlap color to the overlapping areas
    combined_masks = tf.where(tf.cast(overlap_mask, tf.bool), overlap_color, combined_masks)
    logger.debug(f" After where")
    combined_mask_area = tf.math.logical_or(mask1_bool, mask2_bool)
    logger.debug(f" After logical")
    # Convert to float for blending
    combined_mask_area_float = tf.cast(combined_mask_area, tf.float32)
    logger.debug(f" After float")
    # Apply the alpha blending only to areas with masks
    overlayed_image = image * (1 - combined_mask_area_float * alpha) + combined_masks * (combined_mask_area_float * alpha)
    logger.debug(f" After overlay")
    return overlayed_image


def recombine_patches(predictions, original_shapes, patch_counts, patch_size):
    recombined_images = []
    start_index = 0
    
    for (original_height, original_width), patch_count in zip(original_shapes, patch_counts):
        
        # Calculate the number of patches along each dimension
        num_patches_height = int(np.ceil(original_height / patch_size[1]))
        num_patches_width = int(np.ceil(original_width / patch_size[0]))
        #logger.debug(f"calculated patch_size: {(num_patches_height * num_patches_width)}")
        current_patches = predictions[start_index:start_index + num_patches_height * num_patches_width]
        #logger.debug(f"current_patches.shape: {current_patches.shape}")
        start_index += patch_count
        
        # Reshape the patches to align in a grid
        reshaped_patches = tf.reshape(current_patches, [num_patches_height, num_patches_width, patch_size[0], patch_size[1], 1])

        # Rearrange the patches and merge them
        recombined_image = tf.transpose(reshaped_patches, [0, 2, 1, 3, 4])
        recombined_image = tf.reshape(recombined_image, [num_patches_height * patch_size[0], num_patches_width * patch_size[1]])
        logger.debug(f'recombined_image after reshape: {type(recombined_image)}')
        # crop image to remove padding
        recombined_image = recombined_image[:original_height, :original_width]
        logger.debug(f'recombined_image after crop: {type(recombined_image)}')
        # Add to the list of recombined images
        recombined_images.append(recombined_image)
        
    return recombined_images

def close_and_skeletonize(img):
    closed_img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, np.ones(KERNEL_SIZE, np.uint8))
    
    return skeletonize(closed_img)

def calculate_metrics(skeleton_inverted, area_in_mm):
    num_labels, labelled_image, stats, _ = cv2.connectedComponentsWithStats(skeleton_inverted, 8, cv2.CV_32S, connectivity=4)
    areas = stats[:, cv2.CC_STAT_AREA]
    
    feature_counts, three_label_meetings = find_label_meetings(labelled_image, num_labels)
    num_hexagonal = np.count_nonzero(feature_counts == 6)
    return num_labels/area_in_mm, np.std(areas)/np.mean(areas), num_hexagonal/num_labels, feature_counts, three_label_meetings, num_labels, labelled_image

def find_label_meetings(labelled_image, num_labels):
    rows, cols = labelled_image.shape
    meeting_points = []

    num_meetings = np.zeros(num_labels, dtype=int)

    offsets = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]

    for i in range(1, rows - 1):
        for j in range(1, cols - 1):
            if labelled_image[i, j] == 0:
                # Use a set to track unique labels
                unique_labels = set()

                # Check neighbors using precomputed offsets
                for dy, dx in offsets:
                    neighbor_label = labelled_image[i + dy, j + dx]
                    if neighbor_label != 0:
                        unique_labels.add(neighbor_label)

                if len(unique_labels) > 2:
                    meeting_points.append((j, i))
                    for label in unique_labels:
                        num_meetings[label - 1] += 1

    return num_meetings, meeting_points


def visualize_labels(original_image, labelled_image, points):
    logger.debug(f"Starting visualize_labels")
    
    original_image_8bit = np.uint8(255 * original_image)
    num_labels = np.max(labelled_image) + 1  # Including background
    random_colors = np.random.randint(0, 255, size=(num_labels, 3))

    # Create an empty color image for the labels
    colored_labels = np.zeros_like(original_image_8bit)

    # Assign random colors to each label
    for label in range(1, num_labels):  # Skip background
        colored_labels[labelled_image == label] = random_colors[label]

    # Create a mask for non-zero labels
    mask = labelled_image > 0

    # Initialize an empty image for the overlay
    overlayed_image = np.zeros_like(original_image_8bit)

    # Alpha factor (transparency)
    alpha = 0.2

    # Apply the overlay with transparency only on labeled regions
    for i in range(3):  # Loop through color channels
        overlayed_image[:, :, i] = np.where(
            mask,
            np.uint8(alpha * colored_labels[:, :, i] + (1 - alpha) * original_image_8bit[:, :, i]),
            original_image_8bit[:, :, i]
        )
        
        
    for point in points:
        cv2.circle(overlayed_image, point, radius=2, color=(0, 255, 0), thickness=-1)  # Green point
    return overlayed_image