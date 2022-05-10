import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
import os
import tensorflow as tf
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as viz_utils
from object_detection.builders import model_builder
from object_detection.utils import config_util

import cv2 
import numpy as np
from matplotlib import pyplot as plt

CUSTOM_MODEL_NAME = 'my_ssd_mobnet' 
TF_RECORD_SCRIPT_NAME = 'generate_tfrecord.py'
LABEL_MAP_NAME = 'label_map.pbtxt'

paths = {
    'WORKSPACE_PATH': os.path.join('plate_detection', 'workspace'),
    'ANNOTATION_PATH': os.path.join('plate_detection', 'workspace','annotations'),
    'CHECKPOINT_PATH': os.path.join('plate_detection', 'workspace','models',CUSTOM_MODEL_NAME)
 }

files = {
    'PIPELINE_CONFIG':os.path.join('plate_detection', 'workspace','models', CUSTOM_MODEL_NAME, 'pipeline.config'),
    'LABELMAP': os.path.join(paths['ANNOTATION_PATH'], LABEL_MAP_NAME)
}

# Load pipeline config and build a detection model
configs = config_util.get_configs_from_pipeline_file(files['PIPELINE_CONFIG'])
detection_model = model_builder.build(model_config=configs['model'], is_training=False)

# Restore checkpoint
ckpt = tf.compat.v2.train.Checkpoint(model=detection_model)
ckpt.restore(os.path.join(paths['CHECKPOINT_PATH'], 'ckpt-11')).expect_partial()


# Load model architecture, weight and labels
json_file = open('OCR/MobileNets_SSD_character_recognition.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
model = tf.keras.models.model_from_json(loaded_model_json)
model.load_weights("OCR/License_character_recognition.h5")
#print("[INFO] Model loaded successfully...")

labels = LabelEncoder()
labels.classes_ = np.load('OCR/license_character_classes.npy')
#print("[INFO] Labels loaded successfully...")

# pre-processing input images and pedict with model
def predict_from_model(image,model,labels):
    image = cv2.resize(image,(80,80))
    image = np.stack((image,)*3, axis=-1)
    prediction = labels.inverse_transform([np.argmax(model.predict(image[np.newaxis,:]))])
    return prediction

def sort_contours(cnts,reverse = False):
    i = 0
    boundingBoxes = [cv2.boundingRect(c) for c in cnts]
    (cnts, boundingBoxes) = zip(*sorted(zip(cnts, boundingBoxes),
                                        key=lambda b: b[1][i], reverse=reverse))
    return cnts

def get_plate_numbers(image,frame):
    try :
        cropped_image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)

        plate_image = cv2.resize(cropped_image,(140,50))

        blu = cv2.resize(cropped_image,(380,100))
        
        # Applied inversed thresh_binary 
        binary = cv2.threshold(blu, 180, 255,
                            cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

        # Create sort_contours() function to grab the contour of each digit from left to right

        cont, _  = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Initialize a list which will be used to append charater image
        crop_characters = []

        # define standard width and height of character
        digit_w, digit_h = 20, 70

        for c in sort_contours(cont):
            (x, y, w, h) = cv2.boundingRect(c)
            ratio = h/w
            if 1<=ratio<=5.5: # Only select contour with defined ratio
                if h/plate_image.shape[0]>=0.5: # Select contour which has the height larger than 50% of the plate
                    # Sperate number and gibe prediction
                    curr_num = binary[y:y+h,x:x+w]
                    curr_num = cv2.resize(curr_num, dsize=(digit_w, digit_h))
                    _, curr_num = cv2.threshold(curr_num, 220, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                    crop_characters.append(curr_num)

        final_string = ''
    
        for i,character in enumerate(crop_characters):
            title = np.array2string(predict_from_model(character,model,labels))
            final_string+=title.strip("'[]")

        return final_string
        
    except Exception as e:
        print(e)

@tf.function
def detect_fn(image):
    image, shapes = detection_model.preprocess(image)
    prediction_dict = detection_model.predict(image, shapes)
    detections = detection_model.postprocess(prediction_dict, shapes)
    return detections

category_index = label_map_util.create_category_index_from_labelmap(files['LABELMAP'])
#IMAGE_PATH = os.path.join(paths['IMAGE_PATH'], 'Cars381.png')

def get_plate(path):
    try:
        img = cv2.imread(path)
        #img = cv2.resize(img,(600,400))
        image_np = np.array(img)

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
            region = cv2.resize(region , (250,100))
            string = get_plate_numbers(region,img)

        viz_utils.visualize_boxes_and_labels_on_image_array(
                    image_np_with_detections,
                    detections['detection_boxes'],
                    detections['detection_classes']+label_id_offset,
                    detections['detection_scores'],
                    category_index,
                    use_normalized_coordinates=True,
                    max_boxes_to_draw=5,
                    min_score_thresh=.2,
                    agnostic_mode=False)
        return string
    except:
        string = 'no plate number found'
        return string

##    cv2.imshow('hi',image_np_with_detections)
##    cv2.waitKey(0)
##    cv2.imwrite('plate_detected.jpg',image_np_with_detections)

