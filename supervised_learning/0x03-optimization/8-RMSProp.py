#!/usr/bin/env python3
'''training operation for a neural network in tensorflow'''

import tensorflow.compat.v1 as tf


def create_RMSProp_op(loss, alpha, beta2, epsilon):
    '''
    creates training operation for a neural network in tensorflow
    using the RMSProp optimization algorithm.
    :loss: loss of the network
    :alpha: learning rate
    :beta2: RMSProp weight
    :epsilon: small number to avoid division by zero
    returns: RMSProp optimization operation
    '''
    return tf.train.RMSPropOptimizer(learning_rate=alpha,
                                     decay=beta2,
                                     epsilon=epsilon).minimize(loss)
