import tensorflow as tf
from object_detection.utils import visualization_utils as viz_utils
import numpy as np

@tf.function
def detect_fn(image, detection_model):
    """Detect objects in image."""

    image, shapes = detection_model.preprocess(image)
    prediction_dict = detection_model.predict(image, shapes)
    detections = detection_model.postprocess(prediction_dict, shapes)

    return detections, prediction_dict, tf.reshape(shapes, [-1])

def phoneDetection(img, detection_model, category_index):
    """ Detects a phone in an image and draws bounding box around it """
    input_tensor = tf.convert_to_tensor(np.expand_dims(img, 0), dtype=tf.float32)
    detections, predictions_dict, shapes = detect_fn(input_tensor, detection_model)

    label_id_offset = 1
    image_np_with_detections = img.copy()

    _, phone_flag = viz_utils.visualize_boxes_and_labels_on_image_array(
          image_np_with_detections,
          detections['detection_boxes'][0].numpy(),
          (detections['detection_classes'][0].numpy() + label_id_offset).astype(int),
          detections['detection_scores'][0].numpy(),
          category_index,
          use_normalized_coordinates=True,
          max_boxes_to_draw=200,
          min_score_thresh=.30,
          agnostic_mode=False)

    return image_np_with_detections, phone_flag