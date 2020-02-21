import mujoco_py
import gym
import pickle
import os
import json
import time
import brains.continuous_time_rnn as ctrnn

from deap import base
from deap import creator


creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, typecode='b', fitness=creator.FitnessMax)

directory = '2020-02-20_14-32-20'

# Load configuration file
with open(os.path.join('Simulation_Results', 'CTRNN', directory, 'Configuration.json'), "r") as read_file:
    configuration_data = json.load(read_file)

# Load halloffame candidate
with open(os.path.join('Simulation_Results', 'CTRNN', directory, 'HallOfFame.pickle'), "rb") as fp:
    hall_of_fame = pickle.load(fp)

env = gym.make(configuration_data["environment"])

# Set random seed for gym environment
if configuration_data["random_seed_for_environment"] is not -1:
    env.seed(configuration_data["random_seed_for_environment"])

# Get individual from hall of fame
individual = hall_of_fame[0]

# Number of neurons
number_neurons = configuration_data["number_neurons"]
input_size = env.observation_space.shape[0]
output_size = env.action_space.shape[0]

# Alpha
alpha = 0.01

# Size of Individual
IND_SIZE = input_size * number_neurons + number_neurons * number_neurons + number_neurons * output_size

if configuration_data["optimize_y0"]:
    IND_SIZE += number_neurons

env.render()
env.reset()

for i in range(1):

    fitness_current = 0
    ob = env.reset()
    done = False

    # Create Contininuous-Time RNN object
    brain = ctrnn.ContinuousTimeRNN(individual, input_size, number_neurons, output_size,
                                    configuration_data["optimize_y0"],
                                    configuration_data["delta_t"],
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
