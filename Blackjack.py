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
import numpy as np
import random
from datetime import datetime

#Build gym environment
env = gym.make('Blackjack-v1')
env.reset()

done = False

#Open file in specified log location
log_loc = "/home/audrey/Documents/90055_ResearchProject/openAI_sandbox/logs/"
filename = log_loc + "Blackjack-" + datetime.now().strftime("%d-%m-%Y-%H:%M:%S") + ".txt"

log = open(filename, "w")

#Set up Q table for learning agent - it will be an n x m table where n = number of states and m = number of actions
Q = np.zeros([env.observation_space[0].n, env.observation_space[1].n, env.observation_space[2].n, env.action_space.n])

#Total accumulated reward
G = 0

#Learning rate
alpha = 0.8

#Reward discount factor
gamma = 0.9

#Exploration/exploitation constant
epsilon = 0.9

#Number of episodes to train model
num_eps = 1000

#To keep track of wins and losses
wins = 0
losses = 0

for i in range(0, num_eps):
    log.write("Beginning episode: " + str(i) + "\n")
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
            #If the action is not in the env action space, pick randomly again
            if action not in env.action_space:
                action = env.action_space.sample()

        next_state, rew, done, info = env.step(action)

        log.write(str(state) + "," + str(action) + ":" + str(next_state) + "," + str(rew) + "," + str(done) + "," + str(info) + "\n")

        #Update Q table with reward
        #The state will be in the format of a 3-parameter tuple, represesting the current player value, the card shown
        # and whether there are any usable aces in the hand. The last is a boolean, which will have to be converted to
        # an integer to index into the Q table
        current_q = Q[state[0], state[1], int(state[2]), action]
        new_q = current_q + alpha * (rew + gamma * np.max(Q[next_state[0], next_state[1], int(next_state[2])]) - current_q)
        Q[state[0], state[1], int(state[2]), action] = new_q

        state = next_state

        #If you want to show the game as it is being played
        env.render()

    if rew > 0.0:
        wins += 1
        log.write("Win\n")
    else:
        losses += 1
        log.write("Lose\n")

log.write("Total wins = " + str(wins) + "\n")
log.write("Total losses = " + str(losses) + "\n")

#Close the log file
log.close()