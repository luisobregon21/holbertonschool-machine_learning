#!/usr/bin/env python3
"""Run Baum-Welch algorithm"""


import numpy as np


def baum_welch(Observations, Transition, Emission, Initial, iterations=1000):
    """Run Baum-Welch algorithm"""
    if type(Observations) is not np.ndarray or Observations.ndim != 1:
        return None, None
    if type(Emission) is not np.ndarray or Emission.ndim != 2:
        return None, None
    if type(Transition) is not np.ndarray or Transition.ndim != 2:
        return None, None
    if type(Initial) is not np.ndarray or Initial.ndim != 2:
        return None, None
        stateprobs = np.ndarray((N, Observations.shape[0]))
    N = Emission.shape[0]
    if ((Transition.shape[0] != N or Transition.shape[1] != N
         or Initial.shape[0] != N)):
        return None, None
    if type(iterations) is not int or iterations < 1:
        return None, None
    transprev = None
    emprev = None
    initprev = None
    iterations = 1000
    while iterations:
        iterations -= 1
        flike, forward = doforward(Observations, Emission, Transition, Initial)
        blike, backward = dobackward(Observations, Emission,
                                     Transition, Initial)
        forbackxi = forward[:, None, :-1] * backward[None, :, 1:]
        emittedprobs = Emission[:, Observations[1:]]
        xi = forbackxi * Transition[..., None] * emittedprobs[None, :, ...]
        xi /= xi.sum(axis=(0, 1))
        forbackga = forward * backward
        gamma = forbackga / forbackga.sum(axis=0)
        Transition = xi.sum(axis=2) / xi.sum(axis=(1, 2))
        Transition = Transition.T
        for emit in range(Emission.shape[1]):
            gammanum = gamma[:, Observations == emit]
            Emission[:, emit] = gammanum.sum(axis=1) / gamma.sum(axis=1)
        if ((np.all(transprev == Transition)
             and np.all(emprev == Emission)
             and np.all(initprev == Initial))):
            return Transition, Emission
        transprev = Transition
        initprev = Initial
        emprev = Emission
    return Transition, Emission


def doforward(Observation, Emission, Transition, Initial):
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
    for obs in range(Observation.shape[0]):
        state = state * Emission[:, Observation[obs]]
        stateprobs[:, obs] = state
        state = np.matmul(Transition.T, state)
    return stateprobs[:, -1].sum(), stateprobs


def dobackward(Observation, Emission, Transition, Initial):
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
    state = np.asarray([1] * Transition.shape[0])
    stateprobs[:, -1] = state
    for obs in range(Observation.shape[0] - 2, -1, -1):
        state = np.matmul(Transition, state * Emission[:,
                                                       Observation[obs + 1]])
        stateprobs[:, obs] = state
    return (stateprobs[:, 0] * Initial[:, 0]
            * Emission[:, Observation[0]]).sum(), stateprobs
