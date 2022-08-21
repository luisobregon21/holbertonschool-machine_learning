#!/usr/bin/env python3
''' One-Hot Decode '''
import numpy as np


def one_hot_decode(one_hot):
    '''
    converts a one-hot matrix into a vector of labels
    :one_hot: is a one-hot encoded numpy.ndarray with shape (classes, m)
    :Returns: a numpy.ndarray with shape (m,) containing the numeric labels
    for each example, or None on failure
    '''
    if type(one_hot) is np.ndarray and len(one_hot.shape) == 2:
        return np.argmax(one_hot, axis=0)
    return None
