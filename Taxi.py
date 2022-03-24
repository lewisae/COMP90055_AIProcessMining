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
import random

#Build gym environment
env = gym.make('Taxi-v3')
env.reset()

done = False

#Set up Q table for learning agent - it will be an n x m table where n = number of states and m = number of actions
Q = np.zeros([env.observation_space.n, env.action_space.n])

#Total accumulated reward
G = 0

#Learning rate
alpha = 0.8

#Reward discount factor
gamma = 0.9

#Exploration/exploitation constant
epsilon = 0.15

#Number of episodes to train model
num_eps = 100000

for i in range(0, num_eps):
    state = env.reset()

    done = False
    rew = 0.0

    while not done:

        #Check for exploration vs. exploitation agiainst epsilon
        if random.random() < epsilon:
            #Choose a random action
            action = env.action_space.sample()
        else:
            #Trace a path we have done before
            action = np.argmax(Q[state, :])

        next_state, rew, done, info = env.step(action)
        if done and rew < 1:
            rew = -1.0

        #Update Q table with reward
        new_q = Q[state, action] + alpha * (rew + gamma * np.max(Q[next_state]) - Q[state, action])
        Q[state, action] = new_q

        state = next_state

        env.render()

    if rew > 0.0:
        print("Win")
        print(i)
    else:
        print("Lose")
