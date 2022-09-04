#!/usr/bin/env python3
'''
creates a batch normalization layer
for a neural network in tensorflow
'''

import tensorflow.compat.v1 as tf


def create_batch_norm_layer(prev, n, activation):
    '''
    creates a batch normalization layer
    for a neural network in tensorflow.
    :prev: is the activated output of the previous layer.
    :n: is the number of nodes in the layer to be created.
    :activation: is the activation function that should be used on the output
    '''
    init = tf.keras.initializers.VarianceScaling(mode='fan_avg')
    x = tf.keras.layers.Dense(units=n, kernel_initializer=init)
    x_prev = x(prev)
    mean, variance = tf.nn.moments(x_prev, axes=[0])
    scale = tf.Variable(tf.ones([n]), name='gamma')
    offset = tf.Variable(tf.zeros([n]), name='beta')
    epsilon = 1e-8

    normalization = tf.nn.batch_normalization(
        x_prev, mean, variance, offset, scale, epsilon)
    return activation(normalization)
