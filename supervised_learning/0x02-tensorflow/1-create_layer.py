#!/usr/bin/env python3
'''create layer'''

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()


def create_layer(prev, n, activation):
    '''
    creates layer
    :prev: is the activated output of the previous layer
    :n: is the number of nodes in the layer to create
    :activation: is the activation function that the
    layer should use
    '''
    weight = tf.keras.initializers.VarianceScaling(mode='fan_avg')
    layer = tf.keras.layers.Dense(units=n, activation=activation,
                                  kernel_initializer=weight, name='layer')
    return layer(prev)
