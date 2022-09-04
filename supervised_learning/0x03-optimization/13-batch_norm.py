#!/usr/bin/env python3
'''normalizes an unactivated output of a neural network'''

import numpy as np


def batch_norm(Z, gamma, beta, epsilon):
    '''
    normalizes an unactivated output of a neural network
    using batch normalization.
    :Z: is a numpy.ndarray of shape (m, n) that should be normalized.
        m: is the number of data points.
        n: is the number of features in Z.
    :gamma: is a numpy.ndarray of shape (1, n) containing the scales used
    :beta: is a numpy.ndarray of shape (1, n) containing the offsets used
    :epsilon: is a small number used to avoid division by zero.
    '''
    mean = np.mean(Z, axis=0)
    variance = np.var(Z, axis=0)
    Z_norm = (Z - mean) / np.sqrt(variance + epsilon)
    return gamma * Z_norm + beta
