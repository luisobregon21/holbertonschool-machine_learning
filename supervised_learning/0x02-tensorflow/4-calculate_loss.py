#!/usr/bin/env python3
'''create layer'''

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()


def calculate_loss(y, y_pred):
    '''
    calculates the softmax cross-entropy loss of a prediction
    :y: placeholder for the labels of the input data
    :y_pred: tensor containing the networks predictions
    '''
    return tf.losses.softmax_cross_entropy(y, y_pred)
