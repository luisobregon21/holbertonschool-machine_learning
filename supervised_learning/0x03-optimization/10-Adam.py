#!/usr/bin/env python3
'''training operation for a neural network in tensorflow'''

import tensorflow.compat.v1 as tf


def create_Adam_op(loss, alpha, beta1, beta2, epsilon):
    '''
    creates the training operation for a neural network in tensorflow
    using the Adam optimization algorithm
    :loss: is the loss of the network
    :alpha: is the learning rate
    :beta1: is the weight used for the first moment
    :beta2: is the weight used for the second moment
    '''
    return tf.train.AdamOptimizer(alpha, beta1, beta2, epsilon).minimize(loss)
