#!/usr/bin/env python3
'''create layer'''

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()


def create_train_op(loss, alpha):
    '''
    creates the training operation for the network
    :loss: is the loss of the networks prediction
    :alpha: is the learning rate
    '''
    return tf.train.GradientDescentOptimizer(alpha).minimize(loss)
