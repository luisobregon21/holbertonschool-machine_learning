#!/usr/bin/env python3
'''identity Block'''

import tensorflow.keras as K


def identity_block(A_prev, filters):
    '''
    builds an identity block.
    :A_prev: is the output from the previous layer
    :filters: is a tuple or list containing F11,
    F3, F12 respectively:
        F11: is the number of filters in the first 1x1 convolution
        F3: is the number of filters in the 3x3 convolution
        F12: is the number of filters in the second 1x1 convolution
    '''
    init = K.initializers.he_normal()
    activation = 'relu'
    F11, F3, F12 = filters

    conv1 = K.layers.Conv2D(filters=F11, kernel_size=1, padding='same',
                            kernel_initializer=init)(A_prev)
    batch = K.layers.BatchNormalization(axis=3)(conv1)
    activation1 = K.layers.Activation(activation)(batch)
    conv2 = K.layers.Conv2D(filters=F3, kernel_size=3, padding='same',
                            kernel_initializer=init)(activation1)
    batch2 = K.layers.BatchNormalization(axis=3)(conv2)
    activation2 = K.layers.Activation(activation)(batch2)
    conv3 = K.layers.Conv2D(filters=F12, kernel_size=1, padding='same',
                            kernel_initializer=init)(activation2)
    batch3 = K.layers.BatchNormalization(axis=3)(conv3)
    add = K.layers.Add()([batch3, A_prev])
    activation = K.layers.Activation(activation)(add)
    return activation
