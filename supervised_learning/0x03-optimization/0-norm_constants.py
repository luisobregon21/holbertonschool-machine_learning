#!/usr/bin/env python3
'''normalization constants in a matrix'''

import numpy as np


def normalization_constants(X):
    '''
    calculates the normalization (standardization) constants of a matrix
    :X: numpy.ndarray of shape (m, nx) to normalize
    m: is the number of data points
    nx: is the number of features
    '''
    mean = np.mean(X, axis=0)
    stdev = np.std(X, axis=0)
    return mean, stdev
