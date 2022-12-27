#!/usr/bin/env python3
"""marginal."""
import numpy as np
intersection = __import__('1-intersection').intersection


def marginal(x, n, P, Pr):
    """
    Calculate the marginal probability of obtaining the data.

    x (int): number of patients that develop severe side effects
    n (int): total number of patients observed
    P (ndarray): contains the various hypothetical probabilities
    Pr (ndarray): contains the prior beliefs about P
    Returns:
        marginal probability
    """
    if not isinstance(n, int) or (n <= 0):
        raise ValueError("n must be a positive integer")
    if not isinstance(x, int) or (x < 0):
        raise ValueError("x must be an integer that is greater than or equal\
            to 0")
    if x > n:
        raise ValueError("x cannot be greater than n")
    if not isinstance(P, np.ndarray) or len(P.shape) != 1:
        raise TypeError("P must be a 1D numoy.ndarray")
    if not np.all((P >= 0) & (P <= 1)):
        raise ValueError("All values in P must be in the range [0, 1]")
    if not isinstance(Pr, np.ndarray) or (P.shape != Pr.shape):
        raise TypeError("Pr must be a numpy.ndarray with the same shape as P")
    if not np.all((Pr >= 0) & (Pr <= 1)):
        raise ValueError("All values in Pr must be in the range [0, 1]")
    sum = np.sum(Pr)
    if not np.isclose(sum, 1):
        raise ValueError('Pr must sum to 1')
    inter = intersection(x, n, P, Pr)
    return np.sum(inter)
