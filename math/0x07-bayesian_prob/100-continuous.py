#!/usr/bin/env python3
"""posterior"""
import numpy as np
from scipy import special


def posterior(x, n, p1, p2):
    """
    Calculate the posterior probability that the probability of developing\
        severe side effects falls within a specific range given the data.

    x (int): number of patients that develop severe side effects
    n (int): total number of patients observed
    p1 (float): lower bound on the range
    p2 (float): upper bound on the range
    Returns:
        posterior probability that p is within the range
    """
    if (type(n) is not int) or (n <= 0):
        raise ValueError("n must be a positive integer")

    if (type(x) is not int) or (x < 0):
        VE = "x must be an integer that is greater than or equal to 0"
        raise ValueError(VE)

    if x > n:
        raise ValueError("x cannot be greater than n")

    if type(p1) is not float or p1 < 0 or p1 > 1:
        raise ValueError("p1 must be a float in the range [0, 1]")

    if type(p2) is not float or p2 < 0 or p2 > 1:
        raise ValueError("p2 must be a float in the range [0, 1]")

    if p2 <= p1:
        raise ValueError("p2 must be greater than p1")

    b2 = special.btdtr(x + 1, n - x + 1, p2)
    b1 = special.btdtr(x + 1, n - x + 1, p1)
    return b2 - b1
