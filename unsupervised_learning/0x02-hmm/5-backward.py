#!/usr/bin/env python3
"""Perform backward hmm calculation"""


import numpy as np


def backward(Observation, Emission, Transition, Initial):
    """Perform backward hmm calculation"""
    if type(Observation) is not np.ndarray or Observation.ndim != 1:
        return None, None
    if type(Emission) is not np.ndarray or Emission.ndim != 2:
        return None, None
    if type(Transition) is not np.ndarray or Transition.ndim != 2:
        return None, None
    if type(Initial) is not np.ndarray or Initial.ndim != 2:
        return None, None
        stateprobs = np.ndarray((N, Observation.shape[0]))
    N = Emission.shape[0]
    if ((Transition.shape[0] != N or Transition.shape[1] != N
         or Initial.shape[0] != N)):
        return None, None
    stateprobs = np.ndarray((N, Observation.shape[0]))
    state = np.asarray([1] * 5)
    stateprobs[:, -1] = state
    for obs in range(Observation.shape[0] - 2, -1, -1):
        state = np.matmul(Transition, state * Emission[:,
                                                       Observation[obs + 1]])
        stateprobs[:, obs] = state
    return (stateprobs[:, 0] * Initial[:, 0]
            * Emission[:, Observation[0]]).sum(), stateprobs
