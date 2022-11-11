#!/usr/bin/env python3
''' perform PCA on a dataset '''

import numpy as np


def pca(X, ndim):
    '''
    performs PCA on a dataset
    :X: is a numpy.ndarray of shape (n, d) where:
        n: is the number of data points
        d: is the number of dimensions in each point
    :ndim: is the new dimensionality of the transformed X

    Returns: T, a numpy.ndarray of shape (n, ndim)
    containing the transformed version of X
    '''
    X_mean = X - np.mean(X, axis=0)
    u, Sigma, vh = np.linalg.svd(X_mean)
    W = vh.T
    Wr = W[:, :ndim]
    T = np.dot(X_mean, Wr)
    return T
