#!/usr/bin/env python3
''' performs Q-learning '''
import numpy as np


def train(env, Q, episodes=5000, max_steps=100, alpha=0.1, gamma=0.99, epsilon=1, min_epsilon=0.1, epsilon_decay=0.05):
    """
    Perform Q-learning to train the Q-table for the given environment.
    env: the FrozenLake environment instance
    Q: the Q-table as a numpy array
    episodes: the number of episodes to train over
    max_steps: the maximum number of steps per episode
    alpha: the learning rate
    gamma: the discount rate
    epsilon: the initial threshold for epsilon-greedy
    min_epsilon: the minimum value that epsilon should decay to
    epsilon_decay: the decay rate for updating epsilon between episodes
    Returns: Q, total_rewards
    Q: the updated Q-table
    total_rewards: a list containing the rewards per episode
    """
    total_rewards = []
    for episode in range(episodes):
        state = env.reset()
        done = False
        episode_reward = 0

        if type(state) is not int:
            state = state[0]

        for step in range(max_steps):
            # Choose the action with epsilon-greedy
            if np.random.uniform(0, 1) < epsilon:
                action = env.action_space.sample()  # Explore
            else:
                action = np.argmax(Q[state, :])  # Exploit

            # Take the action and observe the next state and reward
            next_state, reward, done, info, prob = env.step(action)
            # Update the Q-table using the Bellman equation
            Q[state, action] = (1 - alpha) * Q[state, action] + alpha * (reward + gamma * np.max(Q[next_state, :]))
            state = next_state
            episode_reward += reward
            if done:
                # Update the reward for falling in a hole
                if reward == 0:
                    episode_reward = -1
                break
        # Decay epsilon
        epsilon = max(min_epsilon, epsilon - epsilon_decay)
        total_rewards.append(episode_reward)
    return Q, total_rewards
