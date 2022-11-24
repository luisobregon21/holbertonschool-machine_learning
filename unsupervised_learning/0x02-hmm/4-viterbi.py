#!/usr/bin/env python3
"""Perform viterbi algorithm"""


import numpy as np


def viterbi(Observation, Emission, Transition, Initial):
    """Perform viterbi algorithm"""
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
    state = Initial
    maxpaths = np.ndarray((N, Observation.shape[0] - 1))
    for obs in range(0, Observation.shape[0]):
        state = state * Emission[:, [Observation[obs]]]
        maxpaths[:, obs - 1] = state.argmax(axis=1)
        state = state.max(axis=1)
        stateprobs[:, obs] = state
        state = Transition.T * state
    lastmaxpath = stateprobs[:, -1].argmax()
    maxpath = [lastmaxpath]
    for obs in range(Observation.shape[0] - 2, -1, -1):
        lastmaxpath = int(maxpaths[lastmaxpath, obs])
        maxpath.append(lastmaxpath)
    return maxpath[::-1], stateprobs[:, -1].max()
