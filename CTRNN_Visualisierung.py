import mujoco_py
import gym
import pickle
import os
import json
import time
import brains.continuous_time_rnn as ctrnn
import matplotlib.pyplot as plt

from deap import base
from deap import creator

creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, typecode='b', fitness=creator.FitnessMax)

directory = '2020-02-23_18-34-48'

# Load configuration file
with open(os.path.join('Simulation_Results', 'CTRNN', directory, 'Configuration.json'), "r") as read_file:
    configuration_data = json.load(read_file)

# Load hall of fame candidates
with open(os.path.join('Simulation_Results', 'CTRNN', directory, 'HallOfFame.pickle'), "rb") as read_file:
    hall_of_fame = pickle.load(read_file)

# Load Log
with open(os.path.join('Simulation_Results', 'CTRNN', directory, 'Log.json'), 'r') as read_file:
    log = json.load(read_file)

# Get individual from hall of fame
individual = hall_of_fame[0]

env = gym.make(configuration_data["environment"])

# Set random seed for gym environment
if configuration_data["random_seed_for_environment"] is not -1:
    env.seed(configuration_data["random_seed_for_environment"])

env.render()
env.reset()

for i in range(1000):

    fitness_current = 0
    ob = env.reset()
    done = False

    # Create Contininuous-Time RNN object
    brain = ctrnn.ContinuousTimeRNN(individual,
                                    env.observation_space.shape[0],
                                    configuration_data["number_neurons"],
                                    env.action_space.shape[0],
                                    configuration_data["optimize_y0"],
                                    configuration_data["delta_t"],
                                    configuration_data["optimize_state_boundaries"],
                                    configuration_data["clipping_range_min"],
                                    configuration_data["clipping_range_max"],
                                    configuration_data["set_principle_diagonal_elements_of_W_negative"])

    # Test fitness through simulation
    while not done:

        # Perform step of the brain simulation
        action = brain.step(ob)

        # Perform simulation step of the environment
        ob, rew, done, info = env.step(action)

        fitness_current += rew

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
