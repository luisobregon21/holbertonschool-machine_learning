#!/usr/bin/env python3
''' has the trained agent play an episoded '''
import numpy as np


def play(env, Q, max_steps=100):
    '''
    :env: is the FrozenLakeEnv instance
    :Q: is a numpy.ndarray containing the Q-table
    :max_steps: is the maximum number of steps in the episode
    Returns: the total rewards for the episode
    '''
    state = env.reset()[0]
    done = False
    total_reward = 0
    step = 0

    while not done and step < max_steps:
        action = np.argmax(Q[state, :])
        state, reward, done, info, prob = env.step(action)
        total_reward += reward
        step += 1
        env.render()

    return total_reward