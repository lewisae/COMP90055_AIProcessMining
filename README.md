Author: Audrey Lewis

COMP90055 Research Project
University of Melbourne, SEM1 2022

* Summary:
This is the repository for the code implemented building and running Q-learning agents using prebuilt Open AI Gym environments. Read more about the Open AI Gym project and the environments contained here: https://gym.openai.com/docs/. The agents will then log information into the "logs" directory locally (unless otherwise specified) in the format (GameName-dd-mm-YY-HH:MM:SS.txt)

The code consists of the following files, with agents built for each of the corresponding environments:
- Frozen Lake: move around a board attempting to reach a goal (+1) without falling into a hole (-1)

- Blackjack: attempt to get as close to 21 with a hand of 2+ cards as possible without going over. 

- Taxi: move around a grid picking up and dropping off passengers at one of four possible locations, with a positive reward (+20) for each correct dropoff, a negative reward (-10) for an incorrect drop off, and a small negative reward (-1) for each timestep taken (incentivising the fastest possible trip)

- Tetris: based on the classic Atari game, 

How to run:
* All of the code for this project was written in Python using version 3.8.10
This code requires the following dependencies - gym, numpy, random, datetime, tensorflow, keras, box2d

* Tetris requires Tensorflow/Keras to build the neural network used by the Deep Q learning agent. The Tetris game also requires the Box2D package to store state information.

* To install the Open AI Gym requirement, run the command "pip install -U gym[all]". 
To install numpy, Tensorflow, Keras, or Box2D, run the command "pip install package name". 
Random and datetime are included with Python install.
