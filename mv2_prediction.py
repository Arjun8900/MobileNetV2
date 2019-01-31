import numpy as np
import sys
import os
import cv2
import tensorflow as tf
import keras
from keras.layers import Dense,GlobalAveragePooling2D
from keras.models import Model
from keras.utils.np_utils  import to_categorical
sys.path.append('/home/arjun/ARJUN/mv2/models/research/slim')
from nets.mobilenet import mobilenet_v2
from keras.applications.mobilenetv2 import MobileNetV2
from datasets import imagenet
from keras.applications import imagenet_utils

model = MobileNetV2(weights='imagenet', include_top=True)


path = 'cat.4059.jpg'   # Replace with custom image path

img = cv2.resize(cv2.imread(path,1), (224,224))
img = np.expand_dims(img, axis=0)
print(img[0])

img = img/255.

print(img[0])

predict = model.predict(img)

results = imagenet_utils.decode_predictions(predict)
print(results)
