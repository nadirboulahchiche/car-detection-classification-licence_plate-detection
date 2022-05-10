import os
import tensorflow as tf
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as viz_utils
from object_detection.builders import model_builder
from object_detection.utils import config_util

import cv2 
import numpy as np
#import get_licence

CUSTOM_MODEL_NAME = 'my_ssd_mobnet' 
TF_RECORD_SCRIPT_NAME = 'generate_tfrecord.py'
LABEL_MAP_NAME = 'label_map.pbtxt'

paths = {
    'WORKSPACE_PATH': os.path.join('car_detection', 'workspace'),
    'ANNOTATION_PATH': os.path.join('car_detection', 'workspace','annotations'),
    'CHECKPOINT_PATH': os.path.join( 'car_detection','workspace','models',CUSTOM_MODEL_NAME)
 }

files = {
    'PIPELINE_CONFIG':os.path.join('car_detection', 'workspace','models', CUSTOM_MODEL_NAME, 'pipeline.config'),
    'LABELMAP': os.path.join(paths['ANNOTATION_PATH'], LABEL_MAP_NAME)
}

# Load pipeline config and build a detection model
configs = config_util.get_configs_from_pipeline_file(files['PIPELINE_CONFIG'])
detection_model = model_builder.build(model_config=configs['model'], is_training=False)

# Restore checkpoint
ckpt = tf.compat.v2.train.Checkpoint(model=detection_model)
ckpt.restore(os.path.join(paths['CHECKPOINT_PATH'], 'ckpt-11')).expect_partial()


def detect_fn(image):
    image, shapes = detection_model.preprocess(image)
    prediction_dict = detection_model.predict(image, shapes)
    detections = detection_model.postprocess(prediction_dict, shapes)
    return detections


category_index = label_map_util.create_category_index_from_labelmap(files['LABELMAP'])
#IMAGE_PATH = os.path.join(paths['IMAGE_PATH'], 'Cars381.png')

def detect_car(frame):
    image_np = np.array(frame)
    
    input_tensor = tf.convert_to_tensor(np.expand_dims(image_np, 0), dtype=tf.float32)
    detections = detect_fn(input_tensor)
    
    num_detections = int(detections.pop('num_detections'))
    detections = {key: value[0, :num_detections].numpy()
                for key, value in detections.items()}
    detections['num_detections'] = num_detections

    # detection_classes should be ints.
    detections['detection_classes'] = detections['detection_classes'].astype(np.int64)

    label_id_offset = 1
    image_np_with_detections = image_np.copy()

    image = image_np_with_detections
    detection_threshold = 0.6
    scores = list(filter(lambda x: x> detection_threshold , detections['detection_scores']))
    boxes = detections['detection_boxes'][:len(scores)]
    classes = detections['detection_classes'][:len(scores)]
    width = image.shape[1]
    height = image.shape[0]


    for idx,box in enumerate(boxes):
        roi = box*[height,width,height,width]
        region = image[int(roi[0]):int(roi[2]),int(roi[1]):int(roi[3])]
        region = cv2.resize(region , (640,480))
        small = cv2.resize(region, (460,260))
        cv2.imwrite('cache_image/small_object.jpg',small)
        cv2.imwrite('cache_image/object.jpg',region)

    # detection_classes should be ints.
    detections['detection_classes'] = detections['detection_classes'].astype(np.int64)

    label_id_offset = 1


    viz_utils.visualize_boxes_and_labels_on_image_array(
                frame,
                detections['detection_boxes'],
                detections['detection_classes']+label_id_offset,
                detections['detection_scores'],
                category_index,
                use_normalized_coordinates=True,
                max_boxes_to_draw=5,
                min_score_thresh=.6,
                agnostic_mode=True)

    return frame

#images = os.listdir('test_images')


def get_car(path):
##    for image in images: 
        frame = cv2.imread(path)
        frame = detect_car(frame) ## this is the detection model for cars
        #
