#!/usr/bin/env python3
''' calculates a correlation matrix '''

import numpy as np


def correlation(C):
    '''
    calculates a correlation matrix
    :C: is a numpy.ndarray of shape (d, d) containing a covariance matrix
        d: is the number of dimensions
    '''
    if type(C) is not np.ndarray:
        raise TypeError('C must be a numpy.ndarray')
    if len(C.shape) != 2 or C.shape[0] != C.shape[1]:
        raise ValueError('C must be a 2D square matrix')

    cov = np.diag(C)
    cov_mat = np.expand_dims(cov, axis=0)
    std_x = np.sqrt(cov_mat)
    std_product = np.dot(std_x.T, std_x)
    corr = C / std_product

    return corr
