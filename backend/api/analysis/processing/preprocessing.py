import tensorflow as tf

@tf.function
def load_image(bytes):
    image = tf.io.decode_image(bytes, channels=3, dtype=tf.float32)
    return image

@tf.function
def load_images_dataset(image_bytes_list):
    # Convert files to a dataset of image contents
    dataset = tf.data.Dataset.from_tensor_slices(image_bytes_list) # dataset of bytes
    # Apply processing to each image
    processed_images = dataset.map(load_image)  # dataset of EagerTensors of shape (None, None, 3)
    return processed_images

@tf.function
def pad_dataset(images_dataset, patch_size, **kwargs):
    # Apply the function to each image and obtain padded images along with original dimensions
    padded_dataset_with_shapes = images_dataset.map(lambda image: apply_padding_and_return_shape(image, patch_size)) # dataset of tuples of (padded_image, original_shape)
    # To extract padded images and their original dimensions separately if needed
    padded_dataset = padded_dataset_with_shapes.map(lambda x, y: x)
    original_shapes = padded_dataset_with_shapes.map(lambda x, y: y)
    return padded_dataset, original_shapes

@tf.function
def apply_padding_and_return_shape(image, patch_size):
    # Record the original dimensions
    original_shape = tf.shape(image)[:2]

    # Calculate padding
    height_pad = (patch_size[0] - original_shape[0] % patch_size[0]) % patch_size[0]
    width_pad = (patch_size[1] - original_shape[1] % patch_size[1]) % patch_size[1]
    padding = [[0, height_pad], [0, width_pad], [0, 0]]

    # Apply padding
    padded_image = tf.pad(image, padding, mode='constant', constant_values=0)
    return padded_image, original_shape

@tf.function
def extract_patches(image, patch_size):
# Extract patches
    patches = tf.image.extract_patches(
        images=tf.expand_dims(image, 0),
        sizes=[1, *patch_size[:2], 1],
        strides=[1, *patch_size[:2], 1],
        rates=[1, 1, 1, 1],
        padding='VALID'
    )
    return tf.reshape(patches, [-1, *patch_size])

@tf.function
def split_images_into_patches(dataset, patch_size):
    # Use TensorFlow's built-in functions for efficient mapping
    dataset = dataset.map(lambda x: extract_patches(x, patch_size))

    patch_counts = dataset.map(lambda x: tf.shape(x)[0])
    
    # Flat map the dataset to have each patch as an element
    dataset = dataset.flat_map(lambda x: tf.data.Dataset.from_tensor_slices(x))

    return dataset, patch_counts

@tf.function
def get_image_dimensions(image):
    return tf.shape(image)[:2]

@tf.function
def resize_with_pad(image, height, width):
  return tf.clip_by_value(tf.image.resize_with_pad(image, height, width, method='lanczos5'), 0., 1.)

@tf.function
def add_dimension(image):
    return tf.expand_dims(image, axis=-1)

@tf.function
def resize_to_next_divisor(image, divisor):
    """
    Resizes an image tensor using tf.image.resize_with_pad so that its dimensions
    are divisible by a specified divisor.

    Parameters:
    image (tf.Tensor): A 3-D image tensor of shape [height, width, channels].
    divisor (int): The divisor number that height and width should be divisible by.

    Returns:
    tf.Tensor: The resized image tensor.
    """
    # Original dimensions
    original_shape = tf.shape(image)[:2]

    # Calculate the target dimensions
    target_height = ((original_shape[0] - 1) // divisor + 1) * divisor
    target_width = ((original_shape[1] - 1) // divisor + 1) * divisor

    # Resize with padding
    resized_image = tf.image.resize_with_pad(image, target_height, target_width)

    return resized_image, original_shape

@tf.function
def round_up(value, factor):
    """Round up the given value to the nearest multiple of the given factor."""
    return value + (-value % factor)

def round_up_image(image, factor):
    """
    Rounds up an image tensor to the nearest multiple of a given factor.

    Parameters:
    image (tf.Tensor): A 3-D image tensor of shape [height, width, channels].
    factor (int): The factor to round up to.

    Returns:
    tf.Tensor: The rounded image tensor.
    """
    # Original dimensions
    original_shape = tf.shape(image)[:2]

    # Calculate the target dimensions
    target_height = round_up(original_shape[0], factor)
    target_width = round_up(original_shape[1], factor)

    # Resize with padding
    resized_image = tf.image.resize_with_pad(image, target_height, target_width)

    return resized_image, original_shape