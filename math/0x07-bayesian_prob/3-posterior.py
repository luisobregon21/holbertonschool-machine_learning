#!/usr/bin/env python3
"""posterior"""
import numpy as np
intersection = __import__('1-intersection').intersection
marginal = __import__('2-marginal').marginal


def posterior(x, n, P, Pr):
    """
    Calculate the posterior probability for the various hypothetical\
        probabilities.

    x (int): number of patients that develop severe side effects
    n (int): total number of patients observed
    P (ndarray): contains the various hypothetical probabilities
    Pr (ndarray): contains the prior beliefs of P
    Returns:
        posterior probability
    """
    return intersection(x, n, P, Pr) / marginal(x, n, P, Pr)
