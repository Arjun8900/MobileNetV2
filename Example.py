
import keras
from keras.utils.np_utils import to_categorical
import numpy as np
import sys
from keras.datasets import cifar100
from keras.applications.mobilenetv2 import MobileNetV2
import tensorflow
import pickle
import os
import cv2
from keras.utils.vis_utils import plot_model
from keras.datasets import cifar100
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Dense,GlobalAveragePooling2D, BatchNormalization
from keras.layers import Conv2D, Reshape, Activation, Dropout
from keras.optimizers import Adam
from keras.models import Model

num_classes = 100
size = 32
batch = 128

# Using my_mv2 script
from my_mv2 import My_Mobilenetv2


(x_train, y_train), (x_test, y_test) = cifar100.load_data(label_mode='fine')
y_train = to_categorical(y_train, num_classes)
y_test = to_categorical(y_test, num_classes)

def unison_shuffled_copies(a, b):
    p = np.random.permutation(len(a))
    return a[p], b[p]
x_train,y_train = unison_shuffled_copies(x_train,y_train)


print(x_train.shape)


model = My_Mobilenetv2((size,size,3), num_classes)

for layer in model.layers:
    layer.trainable = True

model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['accuracy'])

hist = model.fit(x_train, y_train, epochs=40,validation_split=0.05,  shuffle=True, batch_size=batch)
model.save('mv2_cifar100_3.model')




