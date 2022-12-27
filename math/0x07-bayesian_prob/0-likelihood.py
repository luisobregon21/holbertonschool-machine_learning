#!/usr/bin/env python3
""" likelihood."""
import numpy as np


def likelihood(x, n, P):
    """
    Calculate the likelihood of obtaining this data.
    x (int): number of patients that develop severe side effects
    n (int): total number of patients observed
    P (ndarray): contains the various hypothetical probabilities

    Returns:
    ndarray containing the likelihood of obtaining the data
    """
    if not isinstance(n, int) or (n <= 0):
        raise ValueError("n must be a positive integer")
    if not isinstance(x, int) or (x < 0):
        raise ValueError("x must be an integer that is greater than or equal\
            to 0")
    if x > n:
        raise ValueError("x cannot be greater than n")
    if not isinstance(P, np.ndarray) or len(P.shape) != 1:
        raise TypeError("P must be a 1D numpy.ndarray")
    if not np.all((P >= 0) & (P <= 1)):
        raise ValueError("All values in P must be in the range [0, 1]")
    numerator = np.math.factorial(n)
    denominator = (np.math.factorial(x) * (np.math.factorial(n - x)))
    factorial = numerator / denominator
    P_likelihood = factorial * (np.power(P, x)) * (np.power((1 - P), (n - x)))
    return P_likelihood
