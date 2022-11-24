#!/usr/bin/env python3
"""Return steady state probabilities"""


import numpy as np


def regular(P):
    """Return steady state probabilities"""
    if ((type(P) is not np.ndarray or P.ndim != 2 or
         P.shape[0] != P.shape[1] or np.any(P <= 0)
         or not np.all(np.isclose(P.sum(axis=1), 1)))):
        return None
    evals, evecs = np.linalg.eig(P.T)
    evecs = evecs[:, np.isclose(evals, 1)]
    return (evecs / evecs.sum()).T
