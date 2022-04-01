#This file contains all of the logging operations used across the project - each environment will use the same logging
#structure and formatting

from datetime import datetime

#Log location and q-table file location - change depending on system (right now just storing locally)
LOG_LOC = "/home/audrey/Documents/90055_ResearchProject/openAI_sandbox/logs/"
Q_LOC = "/home/audrey/Documents/90055_ResearchProject/openAI_sandbox/q_tables/"


#open_log: opens and returns a txt log file at the specified location (LOG_LOC) for the specified environment name (str)
def open_log(env_name, training=False):
    # Open file in specified log location
    if training:
        env_name += "Training"
    filename = f'{LOG_LOC}{env_name}{datetime.now().strftime("-%Y-%m-%d-T%H:%M:%S")}.txt'
    log = open(filename, "w")
    return log

#open_q: opens and returns a file object to store the Q table for a given env name (str) at the specified location (Q_LOC)
def open_q(env_name):
    q_name = f'{Q_LOC}{env_name}.txt'
    q_file = open(q_name, "w")
    return q_file

#write_log: writes to the given log file in csv format
def write_log(log, episode_index, action):
    current_time = datetime.now().strftime("%Y-%m-%d-T%H:%M:%S.%f")
    log.write(f'{episode_index},{action},{current_time}')

