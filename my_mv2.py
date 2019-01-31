import numpy as np
import keras
from keras.models import Model
from keras.layers import Input, Conv2D, GlobalAveragePooling2D, Dropout
from keras.layers import Activation, BatchNormalization, add, Reshape
from keras.utils.generic_utils import CustomObjectScope

#from keras.applications.mobilenet import relu6  
from keras.layers import DepthwiseConv2D

from keras.utils.vis_utils import plot_model

from keras.utils.generic_utils import CustomObjectScope
from keras.utils.generic_utils import CustomObjectScope
from keras import backend as K

channel_axis=-1
alpha = 1.4
#with CustomObjectScope({'relu6': keras.applications.mobilenet.relu6,'DepthwiseConv2D': keras.applications.mobilenet.DepthwiseConv2D}):

def _make_divisible(v, divisor, min_value=None):
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v
 


def B1(inputs, kernel, filters, t,s):
    #output = inputs.shape[-1] * t
     
    in_filter =int( K.int_shape(inputs)[-1] * t)
    in_conv2d = _make_divisible(in_filter*t, 8)
    out_conv2d = _make_divisible(filters*alpha, 8)
    
    x = Conv2D(in_conv2d, (1,1), padding='same', strides=(1,1))(inputs)
    x = BatchNormalization(axis=-1, momentum=0.999, )(x)
    x = keras.layers.ReLU(6.)(x)
    
    #x = Activation('relu')(x)
    if s==1:
        pad = 'same'
    else:
        pad = 'valid'
    x = DepthwiseConv2D(kernel, strides=(s,s), depth_multiplier=1, padding='same')(x)
    x = BatchNormalization(axis=-1, momentum=0.999)(x)
    x = keras.layers.ReLU(6.)(x)
    
    #x = Activation('relu')(x)
    
    x = Conv2D( out_conv2d, (1,1), strides=(1,1), padding='same')(x)
    x = BatchNormalization(axis=-1, momentum=0.999)(x)
    return x
    
    
def B2(inputs, kernel, filters, t,s):
    #output = inputs.shape[-1] * t
    in_filter = K.int_shape(inputs)[-1] * t
    in_conv2d = _make_divisible(in_filter*t, 8)
    out_conv2d = _make_divisible(filters*alpha, 8)

    x = Conv2D(in_conv2d, (1,1), padding='same', strides=(1,1))(inputs)
    x = BatchNormalization(axis=-1, momentum=0.999)(x)
    x = keras.layers.ReLU(6.)(x)
    
    #x = Activation('relu')(x)
    x = DepthwiseConv2D(kernel, strides=(s,s), depth_multiplier=1, padding='same')(x)
    x = BatchNormalization(axis=channel_axis, momentum=0.999)(x)
    x = keras.layers.ReLU(6.)(x)
    
    #x = Activation('relu')(x)
    
    x = Conv2D(out_conv2d, (1,1), strides=(1,1), padding='same')(x)
    x = BatchNormalization(axis=-1, momentum=0.999)(x)
    
    return add([x, inputs])



def inverted_residual_block(inputs, kernel, filters, t, s,  n):
    # output = B1 + (n-1)*B2
    # s is used by DWC2D
    x = B1(inputs, kernel, filters, t,s)
    for _ in range(1,n):
        x = B2(x, kernel, filters, t,1)
    return x


def My_Mobilenetv2(input_shape, num_classes):
    inputs = Input(shape=input_shape)
    x = Conv2D(32, (3,3), padding='same', strides = (2,2))(inputs)
    x = BatchNormalization(axis=-1, momentum=0.999)(x)
    x = keras.layers.ReLU(6.)(x)
    
    #x = Activation('relu')(x)
    x = inverted_residual_block(x, (3,3), 16, t=1, s=1, n=1)
    x = inverted_residual_block(x, (3,3), 24, t=6, s=2, n=2)
    x = inverted_residual_block(x, (3,3), 32, t=6, s=2, n=3)
    x = inverted_residual_block(x, (3,3), 64, t=6, s=2, n=4)
    x = inverted_residual_block(x, (3,3), 96, t=6, s=1, n=3)
    x = inverted_residual_block(x, (3,3),160, t=6, s=2, n=3)
    x = inverted_residual_block(x, (3,3),320, t=6, s=1, n=1)
    
    x = Conv2D(1280, (1,1), padding='same', strides=(1,1))(x)
    x = BatchNormalization(axis=-1, momentum=0.999)(x)
    x = keras.layers.ReLU(6.)(x)
    
    #x = Activation('relu')(x)

    x = GlobalAveragePooling2D()(x)
    x = keras.layers.Dense(num_classes, activation='softmax', use_bias=True, name='Logits')(x)
    '''
    x = Reshape((1, 1, 1280))(x)
    x = Dropout(0.3, name='Dropout')(x)
    x = Conv2D(num_classes, (1, 1), padding='same')(x)
    x = keras.layers.ReLU(6.)(x)
    
    #x = Activation('softmax', name='softmax')(x)
    output = Reshape((num_classes,))(x)
    '''
    model = Model(inputs, x)
    
    return model


