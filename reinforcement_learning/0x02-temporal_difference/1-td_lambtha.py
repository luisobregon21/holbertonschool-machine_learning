#!/usr/bin/env python3
''' ld_lambtha'''
import numpy as np


def td_lambtha(env, V, policy, lambtha, episodes=5000, max_steps=100, alpha=0.1, gamma=0.99):
    '''
    env is the openAI environment instance
    :V: is a numpy.ndarray of shape (s,) containing the value estimate
    :policy: is a function that takes in a state and returns the next action to take
    :lambtha: is the eligibility trace factor
    :episodes: is the total number of episodes to train over
    :max_steps: is the maximum number of steps per episode
    :alpha: is the learning rate
    :gamma: is the discount rate
    Returns: V, the updated value estimate
    '''
    for i in range(episodes):
        state = env.reset()
        done = False
        E = np.zeros(V.shape)
        for j in range(max_steps):
            action = policy(state)
            next_state, reward, done, info = env.step(action)
            delta = reward + gamma*V[next_state] - V[state]
            E[state] += 1
            for s in range(V.shape[0]):
                V[s] += alpha*delta*E[s]
                E[s] *= gamma*lambtha
            if done:
                break
            state = next_state
    return V
