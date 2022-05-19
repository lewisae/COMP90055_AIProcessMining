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

    for i in Q:
        q_string = str(i).replace('[', '').replace(']', '').replace('\n', '')
        q_file.write(q_string)
        q_file.write("\n")


#This function uses the trained Q table to trace the best possible path - no epsilon greedy strategy
def trained_action(Q, state):
    action = np.argmax(Q[state, :])
    return action

#This function goes through the environment using the trained Q table and logs all of the data
def trained_agent(Q, log, env, env_name, num_eps):
    for i in range(0, num_eps):
        done = False
        state = env.reset()

        while not done:
            action = trained_action(Q, state=state)
            if not env.action_space.contains(action):
                action = env.action_space.sample()
            next_state, rew, done, info = env.step(action)
            logging.write_log(log, i, logging.convert_state(env_name, state))
#            logging.write_log(log, i, logging.convert_state(env_name, state) + "-" + logging.convert_action(env_name, action))
            state = next_state

            # If you want to show the game as it is being played (much slower)
            #env.render()

        #If it is done, we log the end state as the final position
        logging.write_log(log, i, logging.convert_state(env_name, state))
        #logging.write_log(log, i, str(state).replace(", ", ".") + "-" + str(logging.convert_reward(env_name, rew)))
        #logging.write_log(log, i, logging.convert_reward(env_name, rew))