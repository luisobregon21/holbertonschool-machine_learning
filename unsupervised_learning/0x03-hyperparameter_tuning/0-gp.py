#!/usr/bin/env python3
""" Gaussian Process """

import numpy as np


class GaussianProcess:
    """ Class that represents a noiseless 1D Gaussian Process """

    def __init__(self, X_init, Y_init, l=1, sigma_f=1):
        """
        :X_init (ndarray): represents the inputs already sampled with the
        black-box function
        :Y_init (ndarray): represents the outputs of the black-box function
        for each input in X_init
        :l (int, optional): length parameter for the kernel. Defaults to 1.
        :sigma_f (int, optional): standard deviation given to the output of
        the black-box function. Defaults to 1.
        """

        self.X = X_init
        self.Y = Y_init
        self.l = l
        self.sigma_f = sigma_f
        self.K = self.kernel(X_init, X_init)

    def kernel(self, X1, X2):
        """
        Calculate the covariance kernel matrix between two matrices.
        :X1 (ndarray): matrix 1
        :X2 (ndarray): matrix 2
        Returns:
            convariance kernel matrix
        """
        sqdist1 = np.sum(X1 ** 2, 1).reshape(-1, 1) + np.sum(X2 ** 2, 1)
        sqdist2 = 2 * np.dot(X1, X2.T)
        sqdist = sqdist1 - sqdist2
        return self.sigma_f ** 2 * np.exp(-0.5 / self.l ** 2 * sqdist)
