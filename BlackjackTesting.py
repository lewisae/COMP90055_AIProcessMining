'''
COMP90055: Loading an environment and building a Q-learning agent to
The RL environment is built using the Open AI Gym:
https://gym.openai.com/docs/

The code for the Q-learning agent is based off of the following resources:
COMP90054 online textbook: https://gibberblot.github.io/rl-notes/single-agent/model-free.html
Sutton and Barto, Reinforcement Learning: http://incompleteideas.net/book/ebook/
https://www.simplilearn.com/tutorials/machine-learning-tutorial/what-is-q-learning
https://towardsdatascience.com/playing-blackjack-using-model-free-reinforcement-learning-in-google-colab-aa2041a2c13d
https://www.cs.ou.edu/~granville/paper.pdf
'''

import gym
import LogOperations as logging
import QOperations as ops

#Environment is Blackjack
ENV_NAME = "Blackjack"

#Build gym environment
env = gym.make('Blackjack-v1')

#Open logging and q table files
num_eps = 150
log = logging.open_log(ENV_NAME, training=False)
Q = logging.read_q(ENV_NAME)

#Run the trained agent using the provided Q table
ops.trained_agent(Q, log, env, ENV_NAME, num_eps)