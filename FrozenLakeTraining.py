'''
COMP90055: Loading an environment and building a Q-learning agent to
The RL environment is built using the Open AI Gym:
https://gym.openai.com/docs/

The code for the Q-learning agent is based off of the following resources:
COMP90054 online textbook: https://gibberblot.github.io/rl-notes/single-agent/model-free.html
Sutton and Barto, Reinforcement Learning: http://incompleteideas.net/book/ebook/
https://www.simplilearn.com/tutorials/machine-learning-tutorial/what-is-q-learning
https://medium.com/@james_32022/frozen-lake-with-q-learning-4038b804abc1
'''

import gym
import numpy as np
import random
import LogOperations as logging

#Environment is Frozen Lake
ENV_NAME = "FrozenLake"

#Build gym environment
env = gym.make('FrozenLake-v1', is_slippery=False)

#Open logging and q table files
log = logging.open_log(ENV_NAME)
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

#To keep track of wins and losses
wins = 0
losses = 0

def select_action():
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

def update(state, next_state, action, rew):
    # Update Q table with reward
    current = Q[state, action]
    Q[state, action] = current + alpha * ((rew + gamma * np.max(Q[next_state, :])) - current)


for i in range(0, num_eps):
    #Reset the environment for a new episode
    state = env.reset()
    done = False
    rew = 0.0

    #Decaying the epsilon rate as the episodes continue - so we exploit paths we have already taken
    if i % 100 == 0 and epsilon > min_epsilon:
        epsilon *= decay_rate

    while not done:
        #Select an action using epsilon greedy strategy and then execute a step in the environment
        action = select_action()
        next_state, rew, done, info = env.step(action)

        #Negative reinforcement - if we have fallen into a hole, the reward is -1 (to trace good and bad paths)
        if done and rew == 0.0:
            rew = -1.0

        #write log with case/event/action information
        write_log(log, i, action)

        #Update the Q table with the results of this action
        update(state, next_state, action, rew)

        #Move to the next state
        state = next_state

        #If you want to show the game as it is being played (much slower)
        #env.render()


q_file.write(str(Q))

#Close the log file
log.close()
#Close the Q file
q_file.close()
