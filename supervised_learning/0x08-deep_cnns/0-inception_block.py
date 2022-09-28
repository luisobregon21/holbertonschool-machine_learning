#!/usr/bin/env python3
'''Inception Block'''

import tensorflow.keras as K


def inception_block(A_prev, filters):
    '''
    builds an inception block.
    :A_prev: is the output from the previous layer
    :filters: is a tuple or list containing F1, F3R,
    F3,F5R, F5, FPP, respectively:
        F1: is the number of filters in the 1x1 convolution
        F3R: is the number of filters in the 1x1 convolution
        before the 3x3 convolution
        F3: is the number of filters in the 3x3 convolution
        F5R: is the number of filters in the 1x1 convolution
        before the 5x5 convolution
        F5: is the number of filters in the 5x5 convolution
        FPP: is the number of filters in the 1x1 convolution
        after the max pooling (Note : The output shape after
        the max pooling layer is
        outputshape = math.floor((inputshape - 1) / strides) + 1)
    '''
    init = K.initializers.he_normal()
    activation = 'relu'
    F1, F3R, F3, F5R, F5, FPP = filters

    conv1 = K.layers.Conv2D(filters=F1, kernel_size=1, padding='same',
                            activation=activation,
                            kernel_initializer=init)(A_prev)

    conv2 = K.layers.Conv2D(filters=F3R, kernel_size=1, padding='same',
                            activation=activation,
                            kernel_initializer=init)(A_prev)

    conv3 = K.layers.Conv2D(filters=F3, kernel_size=3, padding='same',
                            activation=activation,
                            kernel_initializer=init)(conv2)

    conv4 = K.layers.Conv2D(filters=F5R, kernel_size=1, padding='same',
                            activation=activation,
                            kernel_initializer=init)(A_prev)

    conv5 = K.layers.Conv2D(filters=F5, kernel_size=5, padding='same',
                            activation=activation,
                            kernel_initializer=init)(conv4)

    pooling = K.layers.MaxPooling2D(pool_size=[3, 3], strides=(1, 1),
                                    padding='same')(A_prev)

    layer_poolP = K.layers.Conv2D(filters=FPP, kernel_size=1, padding='same',
                                  activation=activation,
                                  kernel_initializer=init)(pooling)

    mid_layer = K.layers.concatenate([conv1, conv3,
                                      conv5, layer_poolP])

    return mid_layer
