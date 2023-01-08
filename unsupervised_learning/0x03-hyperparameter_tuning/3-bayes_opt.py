#!/usr/bin/env python3
"""BayesianOptimization"""

import numpy as np
GP = __import__('2-gp').GaussianProcess


class BayesianOptimization:
    """Perform Bayesian optimization on a noiseless 1D Gaussian process."""

    def __init__(self, f, X_init, Y_init, bounds, ac_samples, l=1, sigma_f=1,
                 xsi=0.01, minimize=True):
        """
        :f (function): black-box function to be optimized
        :X_init (ndarray): represents the inputs already sampled with the
        black-box function
        :Y_init (ndarray): represents the outputs of the black-box function
        for each input in X_init
        :bounds (tuple): represents the bounds of the space in which to
        look for the optimal point
        :ac_samples (int): number of samples that should be analyzed during
        acquisition
        :l (int, optional): length parameter for the kernel. Defaults to 1.
        :sigma_f (int, optional): standard deviation given to the output of
        the black-box function. Defaults to 1.
        :xsi (float, optional): exploration-exploitation factor for
        acquisition. Defaults to 0.01.
        :minimize (bool, optional): determines whether optimization should
        be performed for minimization or
        maximization. Defaults to True.
        """
        self.f = f
        self.gp = GP(X_init, Y_init, l, sigma_f)
        X_s = np.linspace(bounds[0], bounds[1], num=ac_samples)
        self.X_s = X_s.reshape(-1, 1)
        self.xsi = xsi
        self.minimize = minimize
