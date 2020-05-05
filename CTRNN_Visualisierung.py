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

directory = '2020-05-02_23-07-13'

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

# Set random seed for gym environment
if configuration_data["random_seed_for_environment"] is not -1:
    env.seed(configuration_data["random_seed_for_environment"])

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

for i in range(1):

    fitness_current = 0

    if configuration_data["random_seed_for_environment"] is not -1:
        env.seed(configuration_data["random_seed_for_environment"])
    ob = env.reset()
    done = False

    # Create brain
    brain = brain_class(input_size, output_size, individual, configuration_data)

    # Test fitness through simulation
    while not done:

        # Perform step of the brain simulation
        action = brain.step(ob)

        # Perform simulation step of the environment
        ob, rew, done, info = env.step(action)

        fitness_current += rew

        env.render()

    print(fitness_current)

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
