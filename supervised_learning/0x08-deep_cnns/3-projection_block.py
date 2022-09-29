#!/usr/bin/env python3
'''Projection Block'''
import tensorflow.keras as K


def projection_block(A_prev, filters, s=2):
    '''
    builds a projection block.
    :A_prev: is the output from the previous layer
    :filters: is a tuple or list containing F11, F3, F12, respectively:
        F11: is the number of filters in the first 1x1 convolution
        F3: is the number of filters in the 3x3 convolution
        F12: is the number of filters in the second 1x1 convolution as well
        as the 1x1 convolution in the shortcut connection
    '''
    init = K.initializers.he_normal()
    activation = 'relu'
    F11, F3, F12 = filters

    conv1 = K.layers.Conv2D(filters=F11, kernel_size=1, padding='same',
                            strides=s,
                            kernel_initializer=init)(A_prev)

    batch_norm = K.layers.BatchNormalization(axis=3)(conv1)

    activ = K.layers.Activation(activation)(batch_norm)
    conv2D_1 = K.layers.Conv2D(filters=F3, kernel_size=3, padding='same',
                               kernel_initializer=init)(activ)

    batch_norm1 = K.layers.BatchNormalization(axis=3)(conv2D_1)
    activ_1 = K.layers.Activation(activation)(batch_norm1)
    conv2D_2 = K.layers.Conv2D(filters=F12, kernel_size=1, padding='same',
                               kernel_initializer=init)(activ_1)

    conv2D_3 = K.layers.Conv2D(filters=F12, kernel_size=1, padding='same',
                               strides=s,
                               kernel_initializer=init)(A_prev)
    batch_norm2 = K.layers.BatchNormalization(axis=3)(conv2D_2)
    batch_norm3 = K.layers.BatchNormalization(axis=3)(conv2D_3)

    add = K.layers.Add()([batch_norm2, batch_norm3])
    activation = K.layers.Activation(activation)(add)

    return activation
