#!/usr/bin/env python3
'''sarsa lambtha'''
from numpy import np


def sarsa_lambtha(env, Q, lambtha, episodes=5000, max_steps=100, alpha=0.1, gamma=0.99, epsilon=1, min_epsilon=0.1, epsilon_decay=0.05):
    '''
    :env: is the openAI environment instance
    :Q: is a numpy.ndarray of shape (s,a) containing the Q table
    :lambtha: is the eligibility trace factor
    :episodes: is the total number of episodes to train over
    :max_steps: is the maximum number of steps per episode
    :lpha: is the learning rate
    :gamma: is the discount rate
    :epsilon: is the initial threshold for epsilon greedy
    :min_epsilon: is the minimum value that epsilon should decay to
    :epsilon_decay: is the decay rate for updating epsilon between episodes
    Returns: Q, the updated Q table
    '''
    for i in range(episodes):
        state = env.reset()
        done = False
        action = epsilon_greedy(Q, state, epsilon)
        E = np.zeros(Q.shape)
        for j in range(max_steps):
            next_state, reward, done, info = env.step(action)
            next_action = epsilon_greedy(Q, next_state, epsilon)
            delta = reward + gamma*Q[next_state, next_action] - Q[state, action]
            E[state, action] += 1
            for s in range(Q.shape[0]):
                for a in range(Q.shape[1]):
                    Q[s, a] += alpha*delta*E[s, a]
                    E[s, a] *= gamma*lambtha
            if done:
                break
            state = next_state
            action = next_action
            epsilon = max(min_epsilon, epsilon*(1-epsilon_decay*i))
    return Q

def epsilon_greedy(Q, state, epsilon):
    if np.random.uniform(0, 1) < epsilon:
        action = np.random.randint(Q.shape[1])
    else:
        action = np.argmax(Q[state, :])
    return action
