import mujoco_py
import gym
import pickle
import os
import json
import time
import matplotlib.pyplot as plt
import brains.continuous_time_rnn as ctrnn
import brains.feed_forward as ff
import sys
import numpy as np

from deap import base
from deap import creator

creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, typecode='b', fitness=creator.FitnessMax)

directory = '2020-05-25_10-32-23'
#directory = '2020-05-31_03-14-08'

# Load configuration file
with open(os.path.join('Simulation_Results', directory, 'Configuration.json'), "r") as read_file:
    configuration_data = json.load(read_file)

# Load hall of fame candidates
with open(os.path.join('Simulation_Results', directory, 'HallOfFame.pickle'), "rb") as read_file:
    hall_of_fame = pickle.load(read_file)

# Load Log
with open(os.path.join('Simulation_Results', directory, 'Log.json'), 'r') as read_file:
    log = json.load(read_file)

# Get individual from hall of fame
individual = hall_of_fame[0]

env = gym.make(configuration_data["environment"])

# Get individual size
input_size = env.observation_space.shape[0]
output_size = env.action_space.shape[0]

# Get brain class
if configuration_data["neural_network_type"] == 'LNN':
    brain_class = ff.FeedForwardNN
elif configuration_data["neural_network_type"] == 'CTRNN':
    brain_class = ctrnn.ContinuousTimeRNN
else:
    sys.exit()

env.render()

fitness_complete = 0

N = 500

frames_observation = 20
frames_memory = 20

for i in range(N):

    fitness_current = 0

    if configuration_data["random_seed_for_environment"] is not -1:
        env.seed(configuration_data["random_seed_for_environment"] + i)

    ob = env.reset()
    done = False
    env._max_episode_steps = frames_observation + frames_memory + 50

    # Create brain
    brain = brain_class(input_size, output_size, individual, configuration_data)

    j = 0
    while not done:

        phase = "Observation phase (frames 0-19) - Brain gets input signals for the position of the red object but cannot move the robot arm"

        # Perform step of the brain simulation
        action, neuron_states = brain.step(ob)

        if j <= frames_observation + frames_memory:
            action = np.zeros(output_size)

        # Perform step of the environment simulation
        ob, rew, done, info = env.step(action)

        if j >= frames_observation:
            indices = [4, 5, 8, 9, 10]
            phase = "Memory phase (frames 20-39) - Brain neither gets input signals for the position of the red object nor can move the robot arm"
            for index in indices:
                ob[index] = 0.0

        if j >= frames_observation + frames_memory:
            fitness_current += rew
            phase = "Action phase (frames 40-89) - Brain gets no input signals for the position of the red object but can move the robot arm"

        print("Frame: " + str(j))
        print(phase)
        print("Inputs: " + str(ob.T))
        print("Neuron states: " + str(neuron_states.T))
        print("Outputs: " + str(action.T))
        print("-------------------------------------------------------------------------------------------------------------------------------");

        env.render()
        j += 1

    print(fitness_current)
    fitness_complete += fitness_current

print("Fitness mean: " + str(fitness_complete/N))

# Get statistics from log
generations = [i for i in range(len(log))]
avg = [generation["avg"] for generation in log]
maximum = [generation["max"] for generation in log]

# Plot results
plt.plot(generations, avg, 'r-')
plt.plot(generations, maximum, 'y--')
plt.xlabel('Generations')
plt.legend(['avg', 'std high', 'std low'])
plt.grid()
plt.show()
