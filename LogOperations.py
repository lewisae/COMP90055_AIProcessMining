#This file contains all of the logging operations used across the project - each environment will use the same logging
#structure and formatting

from datetime import datetime, timezone, timedelta

#Log location and q-table file location - change depending on system (right now just storing locally)
import numpy as np

LOG_LOC = "/home/audrey/Documents/90055_ResearchProject/openAI_sandbox/logs/"
Q_LOC = "/home/audrey/Documents/90055_ResearchProject/openAI_sandbox/q_tables/"

BLACKJACK_ACTIONS = {0:"stick", 1:"hit"}
FROZENLAKE_ACTIONS = {0:"up", 1:"down", 2:"right", 3:"left"}
TAXI_ACTIONS = {0:"south", 1:"north", 2:"east", 3:"west", 4:"pickup", 5:"dropoff"}
TETRIS_ACTIONS = {}

#open_log: opens and returns a txt log file at the specified location (LOG_LOC) for the specified environment name (str)
def open_log(env_name, training=False, slippery=False, state_info=False):
    # Open file in specified log location
    if training:
        env_name += "Training"
    if slippery:
        env_name += "Slippery"
    if state_info:
        env_name += "StateInfo"
    filename = f'{LOG_LOC}{env_name}{datetime.now().strftime("-%Y-%m-%d-T%H:%M:%S")}.csv'
    log = open(filename, "w")
    log.write("Index,Action,Timestamp\n")
    return log

#open_q: opens and returns a file object to store the Q table for a given env name (str) at the specified location (Q_LOC)
def open_q(env_name):
    q_name = f'{Q_LOC}{env_name}.txt'
    q_file = open(q_name, "w")
    return q_file

#read_q: opens and reads the Q table, returning a numpy array that can then be used to guide an agent
def read_q(env_name):
    q_name = f'{Q_LOC}{env_name}.txt'
    q_file = np.loadtxt(q_name, dtype=float)
    return q_file


#write_log: writes to the given log file in csv format
def write_log(log, episode_index, action):
    #Get the current time
    current_time = datetime.now(tz=timezone(offset=timedelta(hours=10))).isoformat()
    #Write info (including current timestamp) to the log file
    log.write(f'{episode_index},{action},{current_time}\n')

#convert_action: changes the actions from numerical representation to the words that correspond to those actions
def convert_action(env_name, action):
    if "FrozenLake" in env_name:
        return FROZENLAKE_ACTIONS[action]
    elif "Blackjack" in env_name:
        return BLACKJACK_ACTIONS[action]
    elif "Taxi" in env_name:
        return TAXI_ACTIONS[action]
    else:
        return TETRIS_ACTIONS[action]

#convert_reward: changes the reward from numerical representation to "win" or "lose" response
def convert_reward(env_name, rew):
    if "FrozenLake" in env_name:
        if rew > 0:
            return "win"
        else:
            return "lose"
    elif "Blackjack" in env_name:
        if rew > 0:
            return "win"
        elif rew < 0:
            return "lose"
        else:
            return "draw"
    elif "Taxi" in env_name:
        if rew < 20:
            return "lose"
        else:
            return "win"
    else:
        return str(rew)
