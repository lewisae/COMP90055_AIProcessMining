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
from datetime import datetime

#Build gym environment
env = gym.make('FrozenLake-v1', is_slippery=False)
env.reset()

done = False

#Open file in specified log location
log_loc = "/home/audrey/Documents/90055_ResearchProject/openAI_sandbox/logs/"
filename = log_loc + "FrozenLake-" + datetime.now().strftime("%d-%m-%Y-%H:%M:%S") + ".txt"

log = open(filename, "w")

#Open and store Q-table file
q_loc = "/home/audrey/Documents/90055_ResearchProject/openAI_sandbox/q_tables/"
q_name = q_loc + "FrozenLake" + ".txt"

q_file = open(q_name, "w")

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

for i in range(0, num_eps):
    log.write("Beginning episode: " + str(i) + "\n")
    state = env.reset()

    done = False
    rew = 0.0

    if i % 100 == 0 and epsilon > min_epsilon:
        epsilon *= decay_rate

    while not done:

        #Check for exploration vs. exploitation agiainst epsilon
        if random.random() < epsilon:
            #Choose a random action
            action = env.action_space.sample()
        else:
            #If the Q table is all zeroes (this is for speed of training)
            if np.max(Q[state, :]) > 0:
                action = np.argmax(Q[state, :])
            else:
                action = env.action_space.sample()

        next_state, rew, done, info = env.step(action)

        if done and rew == 0.0:
            rew = -1.0

        log.write(str(state) + "," + str(action) + ":" + str(next_state) + "," + str(rew) + "," + str(done) + "\n")

        #Update Q table with reward
        predict = Q[state, action]
        target = rew + gamma * np.max(Q[next_state, :])
        Q[state, action] = Q[state, action] + alpha * (target - predict)

        state = next_state


        #If you want to show the game as it is being played (much slower)
        #env.render()

    if rew > 0.0:
        wins += 1
        log.write("Win\n")
    else:
        losses += 1
        log.write("Lose\n")

    #To view change over time
    if i%100 == 0:
        log.write(str(wins) + " at " + str(i) + " games\n")
        print(str(wins) + " at " + str(i) + " games\n")

log.write("Total wins = " + str(wins) + "\n")
log.write("Total losses = " + str(losses) + "\n")

q_file.write(str(Q))

#Close the log file
log.close()
#Close the Q file
q_file.close()
