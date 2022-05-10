import tensorflow as tf
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt
import numpy as np
import cv2

model = tf.keras.models.load_model('classification-model/model.h5')
#model.summary()


class_names = ['Golf', 'bmw serie 1', 'chevrolet spark', 'chevroulet aveo', 'clio', 'duster', 'hyundai i10',
               'hyundai tucson', 'logan', 'megane', 'mercedes class a', 'nemo citroen', 'octavia', 'picanto', 'polo', 'sandero', 'seat ibiza',
               'symbol', 'toyota corolla', 'volkswagen tiguan']

#print('all classes : ',class_names)

def rgb2gray(imgs, axs):
    rgb_weights = [0.2989, 0.5870, 0.1140]
    
    return np.expand_dims(np.dot(imgs, rgb_weights), axis=axs)

def get_class(path):
    imgg = image.load_img(path, target_size=(224, 224))

    img = image.img_to_array(imgg)
    img = np.expand_dims(img, axis=0)

    img_pred = model.predict(img)
##    print('Prediction labels: ', img_pred, '\n')

    img_pred_value = np.where(img_pred == np.amax(img_pred))
    #print(img_pred_value[1])

##    plt.imshow(imgg)
    return class_names[img_pred_value[1][0]]
##    plt.show()

##string = get_class('car.jpg')
##print(string)
