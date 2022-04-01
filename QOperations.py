#This file contains all of the Q-table operations used by the FrozenLake, Blackjack, and Taxi learners

import numpy as np
import random

#Using a decaying value of epsilon as i increases, until it reaches a minimum epsilon rate
def decay_epsilon(i, epsilon, min_epsilon, decay_rate):
    if i % 100 == 0 and epsilon > min_epsilon:
        epsilon *= decay_rate

#select_action uses an epsilon-greedy strategy to choose an action
def select_action(env, epsilon, state):
    # Check for exploration vs. exploitation agiainst epsilon
    if random.random() < epsilon:
        # Choose a random action
        action = env.action_space.sample()
    else:
        # If the Q table is all zeroes, choose randomly - else choose max (this is for speed of training),
        if np.max(Q[state, :]) > 0:
            action = np.argmax(Q[state, :])
        else:
            action = env.action_space.sample()
    return action

#update takes in state information and updates the Q table with the new reward value
def update(Q, state, next_state, action, rew, alpha, gamma):
    # Update Q table with reward
    current = Q[state, action]
    Q[state, action] = current + alpha * ((rew + gamma * np.max(Q[next_state, :])) - current)
