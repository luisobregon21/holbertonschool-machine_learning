#!/usr/bin/env python3
"""Module gmm."""

import sklearn.mixture


def gmm(X, k):
    """
    Calculate a GMM from a dataset.

    :X (ndarray): contains the dataset
    :k (int): number of clusters
    Returns:
        pi, m, S, clss, bic
    """
    g = sklearn.mixture.GaussianMixture(n_components=k)
    g.fit(X)
    pi = g.weights_
    m = g.means_
    S = g.covariances_
    clss = g.predict(X)
    bic = g.bic(X)

    return pi, m, S, clss, bic
