'''
COMP90055: Loading an environment and building a Q-learning agent to
The RL environment is built using the Open AI Gym:
https://gym.openai.com/docs/

The code for the Q-learning agent is based off of the following resources:
COMP90054 online textbook: https://gibberblot.github.io/rl-notes/single-agent/model-free.html
Sutton and Barto, Reinforcement Learning: http://incompleteideas.net/book/ebook/
https://www.simplilearn.com/tutorials/machine-learning-tutorial/what-is-q-learning
https://www.learndatasci.com/tutorials/reinforcement-q-learning-scratch-python-openai-gym/

'''

import gym
import numpy as np
import LogOperations as logging
import QOperations as ops

#Environment is Taxi
ENV_NAME = "Taxi"

#Build gym environment
env = gym.make('Taxi-v3')

#Open logging and q table files
log = logging.open_log(ENV_NAME, training=True)
q_file = logging.open_q(ENV_NAME)

#Set up Q table for learning agent - it will be an n x m table where n = number of states and m = number of actions
Q = np.zeros([env.observation_space.n, env.action_space.n])

#Learning rate
alpha = 0.8

#Reward discount factor
gamma = 0.99

#Exploration/exploitation constant
epsilon = 0.1
decay_rate = 0.9
min_epsilon = 0.001

#Number of episodes to train model
num_eps = 10000

ops.train_learner(Q, env, log, q_file, num_eps, epsilon, min_epsilon, decay_rate, alpha, gamma)

#Close the log file
log.close()
#Close the Q file
q_file.close()
