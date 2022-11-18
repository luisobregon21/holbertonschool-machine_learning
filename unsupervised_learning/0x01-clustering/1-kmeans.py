#!/usr/bin/env python3
''' K-means on a data set '''

import numpy as np


def kmeans(X, k, iterations=1000):
    '''
    K-means on a data set
    :X: numpy.ndarray of shape (n, d) containing the dataset
        n: the number of data points
        d: the number of dimensions for each data point
    :k: positive integer containing the number of clusters
    iterations: positive integer containing the maximum number of
    iterations that should be performed
    '''
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None, None
    if not isinstance(k, int) or k <= 0:
        return None, None
    if not isinstance(iterations, int) or iterations <= 0:
        return None, None

    n, d = X.shape
    X_min = X.min(axis=0)
    X_max = X.max(axis=0)

    centroid = np.random.uniform(X_min, X_max, size=(k, d))

    for i in range(iterations):

        centroids = np.copy(centroid)
        centroids_extended = centroid[:, np.newaxis]

        distances = np.sqrt(((X - centroids_extended) ** 2).sum(axis=2))
        clss = np.argmin(distances, axis=0)

        for c in range(k):
            if X[clss == c].size == 0:
                centroid[c] = np.random.uniform(X_min, X_max, size=(1, d))
            else:
                centroid[c] = X[clss == c].mean(axis=0)

        centroids_extended = centroid[:, np.newaxis]
        distances = np.sqrt(((X - centroids_extended) ** 2).sum(axis=2))
        clss = np.argmin(distances, axis=0)

        if (centroids == centroid).all():
            break

    return centroid, clss
