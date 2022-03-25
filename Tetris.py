'''
COMP90055: Loading an environment and building a Q-learning agent to
The RL environment is built using the Open AI Gym:
https://gym.openai.com/docs/

The code for the Q-learning agent is based off of the following resources:
COMP90054 online textbook: https://gibberblot.github.io/rl-notes/single-agent/model-free.html
Sutton and Barto, Reinforcement Learning: http://incompleteideas.net/book/ebook/
https://towardsdatascience.com/deep-q-network-dqn-i-bce08bdf2af
https://towardsdatascience.com/deep-q-learning-for-the-cartpole-44d761085c2f
https://keon.github.io/deep-q-learning/
'''

import gym
import numpy as np
import random
from keras.models import Sequential
from keras.layers import Dense


#Build gym environment - this is using the Discrete version of Lunar Lander
env = gym.make('ALE/Tetris-v5')
env.reset()

done = False

'''
#Set up Q table for learning agent - it will be an n x m table where n = number of states and m = number of actions
Q = np.zeros([env.observation_space.shape[0], env.observation_space.shape[1], env.observation_space.shape[2], env.action_space.n])
'''

#Total accumulated reward
G = 0

#Learning rate
alpha = 0.8

#Reward discount factor
gamma = 0.9

#Exploration/exploitation constant
epsilon = 0.9

#Number of episodes to train model
num_eps = 10000

#Build Sequential NN model using Keras
model = Sequential()

model.add(Dense(24, input_dim=env.observation_space.shape[0]*env.observation_space.shape[1]*env.observation_space.shape[2], activation="relu"))
model.add(Dense(24, activation="relu"))
model.add(Dense(env.action_space.n, activation="linear"))

model.compile(loss="mse", optimizer="adam")

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
            #Trace a path we have done before using the model
            action = np.argmax(model.predict(state)[0])
            #If the action is not in the env action space, pick randomly again
            if action not in env.action_space:
                action = env.action_space.sample()

        next_state, rew, done, info = env.step(action)
        if done and rew < 1:
            rew = -1.0

        #Update Q table with reward
        #The state will be in the format of a 3-parameter tuple, represesting the current player value, the card shown
        # and whether there are any usable aces in the hand. The last is a boolean, which will have to be converted to
        # an integer to index into the Q table
        current_q = Q[state, action]
        new_q = current_q + alpha * (rew + gamma * np.max(Q[next_state[0], next_state[1], next_state[2]]) - current_q)
        Q[state[0], state[1], int(state[2]), action] = new_q

        state = next_state

        env.render()

    if rew > 0.0:
        print("Win")
        print(i)
    else:
        print("Lose")