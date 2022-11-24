#!/usr/bin/env python3
"""Perform forward computation for hidden Markov"""


import numpy as np


def forward(Observation, Emission, Transition, Initial):
    """Perform forward computation for hidden Markov"""
    if type(Observation) is not np.ndarray or Observation.ndim != 1:
        return None, None
    if type(Emission) is not np.ndarray or Emission.ndim != 2:
        return None, None
    if type(Transition) is not np.ndarray or Transition.ndim != 2:
        return None, None
    if type(Initial) is not np.ndarray or Initial.ndim != 2:
        return None, None
    N = Emission.shape[0]
    if ((Transition.shape[0] != N or Transition.shape[1] != N
         or Initial.shape[0] != N)):
        return None, None
    stateprobs = np.ndarray((N, Observation.shape[0]))
    state = Initial[:, 0]
    likely = 1
    for obs in range(Observation.shape[0]):
        likely *= (state[:, np.newaxis] * Emission)[:, Observation[obs]].sum()
        state = state * Emission[:, Observation[obs]]
        stateprobs[:, obs] = state
        state = np.matmul(Transition.T, state)
    return stateprobs[:, -1].sum(), stateprobs
