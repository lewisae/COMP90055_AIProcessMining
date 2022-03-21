'''
COMP90055: Loading an environment and building a Q-learning agent to
The RL environment is built using the Open AI Gym:
https://gym.openai.com/docs/

The code for the Q-learning agent is based off of the following resources:
COMP90054 online textbook: https://gibberblot.github.io/rl-notes/single-agent/model-free.html
https://www.simplilearn.com/tutorials/machine-learning-tutorial/what-is-q-learning
https://www.learndatasci.com/tutorials/reinforcement-q-learning-scratch-python-openai-gym/
'''

import gym
import numpy as np
import random

#Build gym environment
env = gym.make('FrozenLake-v1')
env.reset()

done = False

#Set up Q table for learning agent - it will be an n x m table where n = number of states and m = number of actions
Q = np.zeros([env.observation_space.n, env.action_space.n])

#Total accumulated reward
G = 0

#Learning rate
alpha = 0.1

#Reward discount factor
gamma = 0.6

#Exploration/exploitation constant
epsilon = 0.2

for i in range(0, 100000):
    state = env.reset()

    done = False

    while not done:

        #Check for exploration vs. exploitation agiainst epsilon
        if random.random() < epsilon:
            #Choose a random action
            action = env.action_space.sample()
            print("Explore: " + str(action))
        else:
            #Trace a path we have done before
            action = np.argmax(Q[state])
            print("Exploit:" + str(action))


        next_state, rew, done, info = env.step(action)
        if done and (rew == 0.0):
            rew = -1.0

        #Update Q table with reward
        new_q = Q[state, action] + alpha * (rew + gamma * np.max(Q[next_state]) - Q[state, action])
        Q[state, action] = new_q

        state = next_state
        print(Q)

        env.render()
