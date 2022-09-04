#!/usr/bin/env python3
'''shuffle data in a matrix'''

import numpy as np


def shuffle_data(X, Y):
    '''
    :X: the first numpy.ndarray of shape (m, nx) to shuffle
        m: the number of data points
        nx: the number of features in X
    :Y: the second numpy.ndarray of shape (m, ny) to shuffle
        m: the same number of data points as in X
        ny: the number of features in Y

    '''
    m = X.shape[0]
    shuffle = np.random.permutation(m)
    X_shuffled = X[shuffle]
    Y_shuffled = Y[shuffle]
    return X_shuffled, Y_shuffled
