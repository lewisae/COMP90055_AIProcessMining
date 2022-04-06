#This file contains all of the Q-table operations used by the FrozenLake, Blackjack, and Taxi learners

import numpy as np
import random
import LogOperations as logging

#Using a decaying value of epsilon as i increases, until it reaches a minimum epsilon rate
def decay_epsilon(i, epsilon, min_epsilon, decay_rate):
    if i % 100 == 0 and epsilon > min_epsilon:
        return epsilon * decay_rate
    else:
        return epsilon

#select_action uses an epsilon-greedy strategy to choose an action
def select_action(Q, env, epsilon, state):
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

#This is the bulk of the Q-learner - it will iterate through the number of episodes and generate a Q table for that game,
#finally writing it to a file
def train_learner(Q, env, log, q_file, num_eps, epsilon, min_epsilon, decay_rate, alpha, gamma):
    for i in range(0, num_eps):
        #Reset the environment for a new episode
        state = env.reset()
        done = False
        rew = 0.0

        # Decaying the epsilon rate as the episodes continue - so we exploit paths we have already taken
        epsilon = decay_epsilon(i, epsilon, min_epsilon, decay_rate)

        while not done:
            #Select an action using epsilon greedy strategy and then execute a step in the environment
            #Select an action using epsilon greedy strategy and then execute a step in the environment
            action = select_action(Q, env, epsilon, state)
            if not env.action_space.contains(action):
                action = env.action_space.sample()
            next_state, rew, done, info = env.step(action)

            #Negative reinforcement - if we have fallen into a hole, the reward is -1 (to trace good and bad paths)
            if done and rew == 0.0:
                rew = -1.0

            #write log with case/event/action information
            logging.write_log(log, i, action)

            #Update the Q table with the results of this action
            update(Q, state, next_state, action, rew, alpha, gamma)

            #Move to the next state
            state = next_state

            #If you want to show the game as it is being played (much slower)
            #env.render()
            
    q_file.write(str(Q))
