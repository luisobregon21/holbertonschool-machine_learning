#!/usr/bin/env python3
'''tensorflow layer with l2 regularization'''

import tensorflow.compat.v1 as tf


def l2_reg_create_layer(prev, n, activation, lambtha):
    '''
    creates a tensorflow layer that includes L2 regulazation.
    :prev: a tensor containing the output of the previous layer
    :n: the number of nodes the new layer should contain
    :activation: the activation function that should be used on the layer
    :lambtha: the L2 regularization parameter
    :returns: the output of the new layer
    '''
    init = tf.keras.initializers.VarianceScaling(scale=2.0, mode='fan_avg')
    regularizer = tf.keras.regularizers.L2(lambtha)
    layer = tf.layers.Dense(units=n, activation=activation,
                            kernel_initializer=init,
                            kernel_regularizer=regularizer)
    return layer(prev)
