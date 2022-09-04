#!/usr/bin/env python3
'''normalizes a matrix'''


def normalize(X, m, s):
    '''
    normalizes (standardizes) a matrix
    :X: numpy.ndarray of shape (d, nx) to normalize
        d: is the number of data points
        nx: is the number of features
    :m: numpy.ndarray of shape (nx,) that contains
    the mean of all features of X
    :s: is a numpy.ndarray of shape (nx,) that contains
    the standard deviation of all features of X
    '''

    return (X - m) / s
