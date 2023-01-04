#!/usr/bin/env python3
"""Module BIC"""

import numpy as np
expectation_maximization = __import__('8-EM').expectation_maximization


def BIC(X, kmin=1, kmax=None, iterations=1000, tol=1e-5, verbose=False):
    """
    Find the best number of clusters for a GMM using the Bayesian Information\
    Criterion.

    :X (ndarray): contains the data set
    :kmin (int, optional): contains the minimum number of clusters.
                            Defaults to 1.
    :kmax (int, optional): contains the maximum number of clusters.
                            Defaults to None.
    :iterations (int, optional): contains the maximum number of iterations.
                                Defaults to 1000.
    :tol (float, optional): contains the tolerance. Defaults to 1e-5.
    verbose (bool, optional): determines if the EM algorithm should print
                                information. Defaults to False.
    Returns:
        best_k, best_result, l, b or None, None, None, None
    """
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None, None, None, None
    if type(kmin) != int or kmin <= 0 or kmin >= X.shape[0]:
        return None, None, None, None
    if type(kmax) != int or kmax <= 0 or kmax >= X.shape[0]:
        return None, None, None, None
    if kmin >= kmax:
        return None, None, None, None
    if type(iterations) != int or iterations <= 0:
        return None, None, None, None
    if type(tol) != float or tol <= 0:
        return None, None, None, None
    if type(verbose) != bool:
        return None, None, None, None
    k_best = []
    best_res = []
    logl_val = []
    bic_val = []
    n, d = X.shape
    for k in range(kmin, kmax + 1):
        pi, m, S, _, log_l = expectation_maximization(X, k, iterations, tol,
                                                      verbose)
        k_best.append(k)
        best_res.append((pi, m, S))
        logl_val.append(log_l)
        cov_params = k * d * (d + 1) / 2.
        mean_params = k * d
        p = int(cov_params + mean_params + k - 1)
        bic = p * np.log(n) - 2 * log_l
        bic_val.append(bic)
    bic_val = np.array(bic_val)
    logl_val = np.array(logl_val)
    best_val = np.argmin(bic_val)
    k_best = k_best[best_val]
    best_res = best_res[best_val]
    return k_best, best_res, logl_val, bic_val
