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
Playing Atari with Deep Reinforcement Learning: https://arxiv.org/abs/1312.5602
'''

import random
import gym
import numpy as np
from collections import deque
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Flatten
from rl.memory import SequentialMemory
from rl.agents.dqn import DQNAgent
from rl.policy import EpsGreedyQPolicy

#Build gym environment - this is using the Arcade Learning Environment implementation of Tetris
env = gym.make('ALE/Tetris-v5')

'''
env = gym.make('CartPole-v0')
state = env.reset()
state = np.reshape(state, [1,4])
print(state)
done = False
'''

#Set up a memory buffer using the SequentialMemory option provided for
buffer_len = 1000
#Set a batch size for the replay function
batch_len = 64
#memory = SequentialMemory(limit=buffer_len, window_length=1)
memory = deque(maxlen=buffer_len)

#Learning rate
alpha = 0.01

#Exploration/exploitation constant
epsilon = 0.2

#Reward discount factor
gamma = 0.95

#Number of episodes to train model
num_eps = 10000
time_steps = 5000

#Input and output shapes for the tensorflow model
output_shape = env.action_space.n

#Build Sequential NN model using Keras
model = tf.keras.models.Sequential()
model.add(Flatten())
model.add(tf.keras.layers.Dense(64, activation="relu"))
model.add(tf.keras.layers.Dense(128, activation="relu"))
model.add(tf.keras.layers.Dense(output_shape, activation="linear"))
model.compile(optimizer='sgd', loss='mae')

#using the model to select the next action using an epsilon greedy strategy
def select_action(state):
    if random.random() < epsilon:
        #Choose a random action
        action = env.action_space.sample()
    else:
        prediction = model.predict(state)
        action = np.argmax(prediction[0])
    return action

#replay: this loads the buffer contents into the tensorflow model
def replay():
    batch = random.sample(memory, batch_len)
    for state, action, next_state, rew, done in batch:
        if not done:
            target = rew + gamma*np.amax(model.predict(next_state)[0])
        else:
            target = rew
        current_target = model.predict(state)
        current_target[0][action] = target
        model.fit(state, current_target, verbose=False)

for i in range(0, num_eps):
    state = env.reset()
    #Reshape the state to add the reward dimensions
    state = np.reshape(state, [1, env.observation_space.shape[0], env.observation_space.shape[1], env.observation_space.shape[2]])
    done = False
    t = 0

    while t < time_steps and not done:
        action = select_action(state)

        next_state, rew, done, info = env.step(action)
        #Reshape the next state to add the reward dimensions
        next_state = np.reshape(next_state, [1, env.observation_space.shape[0], env.observation_space.shape[1],
                                   env.observation_space.shape[2]])
        #Add to the memory buffer
        #memory.append((state, action, next_state, rew, done))
        memory.append((state, action, next_state, t, done))

        state = next_state
        t += 1

    #Once it is done, train it with the memory of the episode
    replay()

    #Print out training information
    print(f'Episode {i}/{num_eps}, Score: {t}')