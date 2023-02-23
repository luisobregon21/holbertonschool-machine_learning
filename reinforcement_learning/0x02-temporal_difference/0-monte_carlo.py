#!/usr/bin/env python3
''' monte carlo '''


def monte_carlo(env, V, policy, episodes=5000, max_steps=100, alpha=0.1, gamma=0.99):
  '''
  performs Monte Carlo algorithm
  :env: is the openAI environment instance
  :V: is a numpy.ndarray of shape (s,) containing the value estimate
  :policy: is a function that takes in a state and returns the next action to take
  :episodes: is the total number of episodes to train over
  :max_steps: is the maximum number of steps per episode
  :alpha: is the learning rate
  :gamma: is the discount rate
  Returns: V, the updated value estimate
  '''
  for i in range(episodes):
        state = env.reset()
        done = False
        episode = []
        for j in range(max_steps):
            action = policy(state)
            next_state, reward, done, info = env.step(action)
            episode.append((state, action, reward))
            if done:
                break
            state = next_state
        G = 0
        for t in range(len(episode)-1, -1, -1):
            state, action, reward = episode[t]
            G = gamma*G + reward
            if state not in [x[0] for x in episode[0:t]]:
                V[state] += alpha*(G - V[state])
  return V
