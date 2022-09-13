#!/usr/bin/env python3
''' one-hot marix with Keras'''

import tensorflow.keras as K


def one_hot(labels, classes=None):
    '''
    converts a label vector into a one-hot matrix
    :labels: is a numpy.ndarray with shape (m,) containing
    numeric class labels
    :classes: is the maximum number of classes
    :return: the one-hot matrix
    '''
    return K.utils.to_categorical(labels, num_classes=classes)
