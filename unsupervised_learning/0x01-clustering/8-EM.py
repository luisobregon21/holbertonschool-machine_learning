#!/usr/bin/env python3
"""Module expectation_maximization."""

import numpy as np
initialize = __import__('4-initialize').initialize
expectation = __import__('6-expectation').expectation
maximization = __import__('7-maximization').maximization


def expectation_maximization(X, k, iterations=1000, tol=1e-5, verbose=False):
    """
    Perform the expectation maximization for a GMM.

    :X (ndarray): contains the data set
    :k (int): contains the number of clusters
    :iterations (int, optional): contains the maximum number of iterations.
                                Defaults to 1000.
    :tol (float, optional): contains tolerance of the log likelihood, used
                            to determine early stopping. Defaults to 1e-5.
    :verbose (bool, optional): determines if you should print information.
                                  Defaults to False.
    Returns:
        pi, m, S, g, l or None, None, None, None, None
    """
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None, None, None, None, None
    if type(k) != int or k <= 0 or X.shape[0] < k:
        return None, None, None, None, None
    if type(iterations) != int or iterations <= 0:
        return None, None, None, None, None
    if type(tol) != float or tol < 0:
        return None, None, None, None, None
    if type(verbose) != bool:
        return None, None, None, None, None
    pi, m, S = initialize(X, k)
    loglikelihood = 0
    i = 0
    while i < iterations:
        g, loglikelihood_new = expectation(X, pi, m, S)
        if verbose is True and (i % 10 == 0):
            print("Log Likelihood after {} iterations: {}".
                  format(i, loglikelihood_new.round(5)))
        if abs(loglikelihood_new - loglikelihood) <= tol:
            break
        pi, m, S = maximization(X, g)
        i += 1
        loglikelihood = loglikelihood_new
    g, loglikelihood_new = expectation(X, pi, m, S)
    if verbose is True:
        print("Log Likelihood after {} iterations: {}".
              format(i, loglikelihood_new.round(5)))
    return pi, m, S, g, loglikelihood_new
