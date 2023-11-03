from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework.parsers import MultiPartParser
from django.http import HttpResponse
from rest_framework import status
import tensorflow as tf
# import preprocessing functions
import io
from PIL import Image
import zipfile
from typing import Tuple
import numpy as np

def process_image(image_content):
    image = tf.image.decode_png(image_content, channels=3)
    image = tf.image.convert_image_dtype(image, tf.float32)

    return image

@tf.function
def split_image(image: tf.Tensor, mask: tf.Tensor = None, patch_size: Tuple[int, int, int] = (128, 128, 3)) -> tf.data.Dataset:
    """
    Split an image (and optionally a mask) into smaller patches of size `patch_size`, padding if necessary.

    Parameters:
    - image (tf.Tensor): The image tensor.
    - mask (tf.Tensor, optional): The mask tensor.
    - patch_size (Tuple[int, int, int]): The size of the patches to split the image into.

    Returns:
    - tf.data.Dataset: A TensorFlow dataset containing image-mask pairs or just images.
    """
    # Calculate padding size
    height_pad = tf.maximum(patch_size[0] - tf.shape(image)[0] % patch_size[0], 0)
    width_pad = tf.maximum(patch_size[1] - tf.shape(image)[1] % patch_size[1], 0)
    padding = [[0, height_pad], [0, width_pad], [0, 0]]

    # Pad image (and mask if provided)
    padded_image = tf.pad(image, padding, mode='constant', constant_values=0)
    patches_image = tf.image.extract_patches(
        images=tf.expand_dims(padded_image, 0),
        sizes=[1, *patch_size[:2], 1],
        strides=[1, *patch_size[:2], 1],
        rates=[1, 1, 1, 1],
        padding='VALID'
    )
    patches_image = tf.reshape(patches_image, [-1, *patch_size])

    if mask is not None:
        padded_mask = tf.pad(mask, padding, mode='constant', constant_values=0)
        patches_mask = tf.image.extract_patches(
            images=tf.expand_dims(padded_mask, 0),
            sizes=[1, *patch_size[:2], 1],
            strides=[1, *patch_size[:2], 1],
            rates=[1, 1, 1, 1],
            padding='VALID'
        )
        patches_mask = tf.reshape(patches_mask, [-1, *patch_size[:2], 1])
        return tf.data.Dataset.from_tensor_slices((patches_image, patches_mask))

    return tf.data.Dataset.from_tensor_slices(patches_image)

model = tf.keras.models.load_model('api/analysis/models/reference30')

class AnalysisView(APIView):
    parser_classes = (MultiPartParser,)
    def post(self, request, *args, **kwargs):
        files = request.FILES.getlist('files')
        if not files:
            return Response('No files provided', status=status.HTTP_400_BAD_REQUEST)

        # Convert files to a dataset of image contents
        image_contents = [file.read() for file in files]
        dataset = tf.data.Dataset.from_tensor_slices(image_contents)

        # Apply processing to each image
        processed_images = dataset.map(process_image)
        processed_images = processed_images.flat_map(lambda img: split_image(img))
        processed_images = processed_images.batch(32).prefetch(tf.data.AUTOTUNE)
        
        threshold = 0.15
        predictions = model.predict(processed_images)

        # Convert predictions to binary images and save them to a ZIP archive
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
            for i, prediction in enumerate(predictions):
                # Convert to binary array, and then to uint8
                binary_prediction = (prediction > threshold).astype(np.uint8) * 255
                
                # Remove the channels dimension if it's 1 (i.e., for grayscale images)
                if binary_prediction.shape[-1] == 1:
                    binary_prediction = binary_prediction[..., 0]
                
                # Convert NumPy array to image
                image = Image.fromarray(binary_prediction)
                
                # Save image to bytes buffer
                image_buffer = io.BytesIO()
                image.save(image_buffer, format='PNG')
                
                # Add image to ZIP archive
                zip_file.writestr(f'image_{i}.png', image_buffer.getvalue())

        # Create HttpResponse with ZIP file data
        response = HttpResponse(buffer.getvalue(), content_type='application/zip')
        response['Content-Disposition'] = 'attachment; filename=images.zip'
        
        return response

    def get(self, request):
        # Here, you would need to decide which image to send back if multiple images are saved
        try:
            with open('test/temp_image_1.png', 'rb') as f:
                return HttpResponse(f.read(), content_type="image/png")
        except IOError as e:
            # get message from exception object
            print(e)
            return Response({'error': 'File not found'}, status=404)