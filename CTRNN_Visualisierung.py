
import gym
import pickle
import os
import json
import time
import matplotlib.pyplot as plt
import brains.continuous_time_rnn as ctrnn
import brains.layered_nn as lnn
import sys
import argparse
import numpy as np

from deap import base
from deap import creator

parser = argparse.ArgumentParser(description='Visualize experiment results')
parser.add_argument('--dir', metavar='dir', type=str, help='directory', default='2020-03-09_17-48-07')
parser.add_argument('--type', metavar='type', type=str, help='directory', default='CTRNN')
parser.add_argument('--hof', metavar='type', type=int, help='how many HallOfFame Individuals should be displayed?', default='1')
args = parser.parse_args()
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, typecode='b', fitness=creator.FitnessMax)

directory = args.dir
experiment_type = args.type


# Load configuration file
with open(os.path.join('Simulation_Results', experiment_type, directory, 'Configuration.json'), "r") as read_file:
    configuration_data = json.load(read_file)

# Load hall of fame candidates
with open(os.path.join('Simulation_Results', experiment_type, directory, 'HallOfFame.pickle'), "rb") as read_file:
    hall_of_fame = pickle.load(read_file)

# Load Log
with open(os.path.join('Simulation_Results', experiment_type, directory, 'Log.json'), 'r') as read_file:
    log = json.load(read_file)

env = gym.make(configuration_data["environment"])

# Set random seed for gym environment
if configuration_data["random_seed_for_environment"] is not -1:
    env.seed(configuration_data["random_seed_for_environment"])

# Get individual size
input_size = env.observation_space.shape[0]
if env.action_space.shape:
    output_size = env.action_space.shape[0]
    discrete_actions = False
else:
    output_size = env.action_space.n
    # todo: make discrete actions an explicit option in configuratio
    discrete_actions = True

# Get brain class
if configuration_data["neural_network_type"] == 'LNN':
    brain_class = lnn.LayeredNN
elif configuration_data["neural_network_type"] == 'CTRNN':
    brain_class = ctrnn.ContinuousTimeRNN
else:
    sys.exit()

env.render()
env.reset()

for individual in hall_of_fame[:args.hof]:

    fitness_current = 0
    ob = env.reset()
    done = False

    # Create brain
    brain = brain_class(input_size, output_size, individual, configuration_data)
    consecutive_non_movement = 0
    # Test fitness through simulation
    while not done:
        # Perform step of the brain simulation
        action = brain.step(ob)

        # Perform simulation step of the environment
        if discrete_actions:
            action = np.argmax(action)
        ob, rew, done, info = env.step(action)

        fitness_current += rew

        if configuration_data["environment"] == "BipedalWalker-v3":
            if ob[2] < 0.0001:
                consecutive_non_movement = consecutive_non_movement + 1
                if consecutive_non_movement > 50:
                    done = True
                    print("aborting because stagnation")
                consecutive_non_movement = 0

        env.render()
        time.sleep(0.01)

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
