#!/usr/bin/env python3
""" Markov_chain. """
import numpy as np


def markov_chain(P, s, t=1):
    """
    Determine the probability of a markov chain being in a particular state.
    :P: (ndarray): represents the transition matrix
    :s: (ndarray): represents the probability of starting in each state
    :t: (int, optional): number of iterations that the markov chain has been
                        through. Defaults to 1.
    """
    if ((type(P) is not np.ndarray or type(s) is not np.ndarray
         or P.ndim != 2 or s.ndim != 2 or P.shape[0] != P.shape[1]
         or s.shape[0] != 1 or s.shape[1] != P.shape[0]
         or type(t) is not int or t < 0)):
        return None
    return np.matmul(s, np.linalg.matrix_power(P, t))
