import numpy as np
import tensorflow as tf
import logging
import cv2
from skimage.morphology import skeletonize

logger = logging.getLogger(__name__)


def overlay_masks(image, reference, prediction, reference_color = (1, 0, 0), prediction_color = (1, 1, 0), overlap_color = (0, 1, 0), alpha=0.6):
    logger.debug(f"Original image shape: {image.shape}")
    logger.debug(f"Reference shape: {reference.shape}")
    logger.debug(f"Prediction shape: {prediction.shape}")
    
    # Convert float masks to boolean
    mask1_bool = reference > 0.5
    mask2_bool = prediction > 0

    #logger.debug(f" After bool")
    # Reshape masks to 3D (x, y, 1)
    mask1_3d = tf.cast(mask1_bool, tf.float32)
    mask2_3d = tf.cast(mask2_bool, tf.float32)

    #logger.debug(f" After 3d")
    # Apply colors to masks
    mask1_colored = mask1_3d * reference_color
    mask2_colored = mask2_3d * prediction_color
    #logger.debug(f" After colored")
    # Determine overlapping area and apply overlap color
    overlap_mask = tf.math.logical_and(mask1_bool, mask2_bool)    

    #overlap_colored = tf.cast(overlap_mask, tf.float32) * overlap_color
    #logger.debug(f" After overlap")
    # Combine masks
    combined_masks = mask1_colored + mask2_colored
    #logger.debug(f" After combine")
    # Apply overlap color to the overlapping areas
    combined_masks = tf.where(tf.cast(overlap_mask, tf.bool), overlap_color, combined_masks)
    #logger.debug(f" After where")
    combined_mask_area = tf.math.logical_or(mask1_bool, mask2_bool)
    #logger.debug(f" After logical")
    # Convert to float for blending
    combined_mask_area_float = tf.cast(combined_mask_area, tf.float32)
    #logger.debug(f" After float")
    # Apply the alpha blending only to areas with masks
    overlayed_image = image * (1 - combined_mask_area_float * alpha) + combined_masks * (combined_mask_area_float * alpha)
    #logger.debug(f" After overlay")
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
        recombined_image = tf.reshape(recombined_image, [num_patches_height * patch_size[0], num_patches_width * patch_size[1], 1])
        #logger.debug(f'recombined_image type after reshaping: {type(recombined_image)}')
        # crop image to remove padding
        recombined_image = recombined_image[:original_height, :original_width]
        #logger.debug(f'recombined_image type after cropping: {type(recombined_image)}')
        # Add to the list of recombined images
        recombined_images.append(recombined_image)
        
    return recombined_images

def close_and_skeletonize(img):
    KERNEL_SIZE = (3,3)
    closed_img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, np.ones(KERNEL_SIZE, np.uint8))
    
    return skeletonize(closed_img)

def calculate_metrics(skeleton_inverted, area_per_pixel):
    num_labels, labelled_image, stats, _ = cv2.connectedComponentsWithStats(skeleton_inverted, 8, cv2.CV_32S, connectivity=4)
    num_labels = num_labels - 1             # Ignore background label
    areas = stats[1:, cv2.CC_STAT_AREA]     # Ignore background label
    
    #logger.debug(f"sum(areas): {np.sum(areas)}")
    #logger.debug(f"max(areas): {np.max(areas)}")
    
    feature_counts, three_label_meetings = find_label_meetings(labelled_image, num_labels)
    num_hexagonal = np.count_nonzero(feature_counts == 6)
    #logger.debug(f"num_hexagonal: {num_hexagonal}")
    
    #logger.debug(f"num_labels: {num_labels}")
    size = skeleton_inverted.size
    #logger.debug(f"size: {size}")
    
    area = size * area_per_pixel
    #logger.debug(f"area: {area}")
    
    cell_density = num_labels/area
    #logger.debug(f"cell_density: {cell_density}")
    
    std_areas = np.std(areas)
    #logger.debug(f"std_areas: {std_areas}")
    
    mean_areas = np.mean(areas)
    #logger.debug(f"mean_areas: {mean_areas}")
    
    coefficient_value = std_areas / mean_areas
    #logger.debug(f"coefficient_value: {coefficient_value}")
    
    hexagonal_cell_ratio = num_hexagonal / num_labels * 100
    #logger.debug(f"hexagonal_cell_ratio: {hexagonal_cell_ratio}")
    
    return num_labels, area, cell_density, std_areas, mean_areas, coefficient_value, num_hexagonal, hexagonal_cell_ratio, feature_counts, three_label_meetings, num_labels, labelled_image

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


@tf.function
def calculate_padding_and_resize(image, resized_height, resized_width, original_height, original_width):
    # Convert all input dimensions to float for consistency in calculations
    resized_height = tf.cast(resized_height, tf.float32)
    resized_width = tf.cast(resized_width, tf.float32)
    original_height = tf.cast(original_height, tf.float32)
    original_width = tf.cast(original_width, tf.float32)
    
    # Calculate aspect ratios
    original_aspect_ratio = original_width / original_height
    resized_aspect_ratio = resized_width / resized_height

    # Initialize padding values as floats
    pad_h, pad_w = tf.constant(0.0, dtype=tf.float32), tf.constant(0.0, dtype=tf.float32)

    # Calculate padding based on aspect ratios
    if original_aspect_ratio > resized_aspect_ratio:
        # Padding was added vertically
        new_height = resized_width / original_aspect_ratio
        pad_h = (resized_height - new_height) / 2
    else:
        # Padding was added horizontally
        new_width = resized_height * original_aspect_ratio
        pad_w = (resized_width - new_width) / 2

    # Convert padding to integer for tf.image.crop_to_bounding_box function
    pad_h_int = tf.cast(pad_h, tf.int32)
    pad_w_int = tf.cast(pad_w, tf.int32)

    # Crop out the padding to restore the original aspect ratio
    cropped_image = tf.image.crop_to_bounding_box(image, pad_h_int, pad_w_int, tf.cast(resized_height - 2 * pad_h, tf.int32), tf.cast(resized_width - 2 * pad_w, tf.int32))

    # Resize the image back to its original dimensions
    resized_image = tf.image.resize(cropped_image, [original_height, original_width], method='nearest')

    logger.debug(f"binarized_image: {resized_image}")
    return resized_image
    

@tf.function
def remove_dimension(images_dataset):
    return images_dataset.map(lambda image: tf.squeeze(image, axis=0))