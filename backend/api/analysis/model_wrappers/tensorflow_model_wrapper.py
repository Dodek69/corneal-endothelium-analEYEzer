from .abstract_model_wrapper import AbstractModelWrapper
import tensorflow as tf
import logging

from tensorflow import keras

@keras.saving.register_keras_serializable()
def tversky_loss(y_true, y_pred, alpha=0.5, beta=0.5, smooth=1):
    # Flatten the input to 1D.
    y_true_f = tf.cast(tf.reshape(y_true, [-1]), tf.float32)
    y_pred_f = tf.cast(tf.reshape(y_pred, [-1]), tf.float32)
    
    # True positives, false positives, and false negatives.
    true_pos = tf.reduce_sum(y_true_f * y_pred_f)
    false_neg = tf.reduce_sum(y_true_f * (1 - y_pred_f))
    false_pos = tf.reduce_sum((1 - y_true_f) * y_pred_f)
    
    # Calculate the Tversky index.
    tversky_index = (true_pos + smooth) / (true_pos + alpha * false_neg + beta * false_pos + smooth)
    
    # Return Tversky loss.
    return 1 - tversky_index
    
@keras.saving.register_keras_serializable()
def dice_loss(y_true, y_pred, smooth=1):
    return tversky_loss(y_true=y_true, y_pred=y_pred, alpha=0.5, beta=0.5, smooth=smooth)

@keras.saving.register_keras_serializable()
def iou(y_true, y_pred, smooth=1):
    
    y_true = tf.cast(tf.reshape(y_true, [-1]), tf.float32)
    y_pred = tf.cast(tf.reshape(y_pred, [-1]), tf.float32)
    
    # calculate intersection and union
    intersection = tf.reduce_sum(y_true * y_pred)
    union = tf.reduce_sum(y_true) + tf.reduce_sum(y_pred) - intersection
    # calculate IoU
    iou = (intersection + smooth) / (union + smooth)
    # return loss
    return 1 - iou

@keras.saving.register_keras_serializable()
def iou_loss(y_true, y_pred, smooth=1):
    return 1 - iou(y_true, y_pred, smooth=smooth)

logger = logging.getLogger(__name__)

class TensorFlowModelWrapper(AbstractModelWrapper):
    def __init__(self, model_path):
        self.model_path = model_path
        
    def load_model(self, model_path=None):
        load_from = model_path if model_path else self.model_path
        logger.info(f"Loading model from {load_from}...")
        self.model = tf.keras.models.load_model(load_from)
        logger.info(f"Loaded model input shape: {self.model.input_shape}, output shape: {self.model.output_shape}")
        return self

    def predict(self, input_data):
        return self.model.predict(input_data)