#!/usr/bin/env python3
''' initializes cluster centroids for K-means '''

import numpy as np


def initialize(X, k):
    '''
    Initializes cluster centroids for K-means
    :X: numpy.ndarray of shape (n, d) containing the dataset
           n: the number of data points
           d: the number of dimensions for each data point
    :k: positive integer containing the number of clusters
    '''

    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None
    if not isinstance(k, int) or k <= 0:
        return None

    n, d = X.shape

    init_centroids = np.random.uniform(
        X.min(axis=0),
        X.max(axis=0),
        size=(k, d)
        )
    return init_centroids
