#!/usr/bin/env python3
"""Module maximization."""

import numpy as np


def maximization(X, g):
    """
    Calculate the maximization step in the EM algorithm for a GMM.

    :X (ndarray): contains the data set
    :g (ndarray): contains the posterior probabilities
    Returns:
        pi, m, S or None, None, None
    """
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None, None, None
    if not isinstance(g, np.ndarray) or len(X.shape) != 2:
        return None, None, None
    gaussian_components = g
    k = gaussian_components.shape[0]
    n, d = X.shape
    posterior_prob = np.sum(gaussian_components, axis=0)
    check = np.sum(posterior_prob)
    if check != X.shape[0]:
        return None, None, None
    priors = np.zeros((k,))
    centroid_updated = np.zeros((k, d))
    covariance_updated = np.zeros((k, d, d))
    for i in range(k):
        mu_up = np.sum((gaussian_components[i, :, np.newaxis] * X), axis=0)
        mu_down = np.sum(gaussian_components[i], axis=0)
        centroid_updated[i] = mu_up / mu_down
        x_m = X - centroid_updated[i]
        sigma_up = np.matmul(gaussian_components[i] * x_m.T, x_m)
        sigma_down = np.sum(gaussian_components[i])
        covariance_updated[i] = sigma_up / sigma_down
        priors[i] = np.sum(gaussian_components[i]) / n
    return priors, centroid_updated, covariance_updated
