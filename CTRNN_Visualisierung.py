import mujoco_py
import gym
import pickle
import os
import json

import numpy as np
import time

from deap import base
from deap import creator


def continuous_time_rnn(y, alpha, V, W, u):

    u2 = u[:, np.newaxis]

    # Differential equation
    dydt = np.dot(W, np.tanh(y)) + np.dot(V, u2)

    return dydt


creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, typecode='b', fitness=creator.FitnessMax)

directory = '2020-02-20_13-30-01'

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

# Number of neurons
number_neurons = configuration_data["number_neurons"]
input_size = env.observation_space.shape[0]
output_size = env.action_space.shape[0]

# Alpha
alpha = 0.01

delta_t = configuration_data["delta_t"]

# Size of Individual
IND_SIZE = input_size * number_neurons + number_neurons * number_neurons + number_neurons * output_size

if configuration_data["optimize_y0"]:
    IND_SIZE += number_neurons

env.render()
env.reset()

fitness_current = 0
done = False

number_steps = 0

V_size = input_size * number_neurons
W_size = number_neurons * number_neurons
T_size = number_neurons * output_size

individual = hall_of_fame[0]

# Get weight matrizes of current individual
V = np.array([[element] for element in individual[0:V_size]])
W = np.array([[element] for element in individual[V_size:V_size + W_size]])
T = np.array([[element] for element in individual[V_size + W_size:V_size + W_size + T_size]])

V = V.reshape([number_neurons, input_size])
W = W.reshape([number_neurons, number_neurons])
T = T.reshape([number_neurons, output_size])

# Set elements of main diagonal to less than 0
if configuration_data["set_priciple_diagonal_elements_of_W_negative"]:
    for j in range(number_neurons):
        W[j][j] = -abs(W[j][j])

for i in range(1):

    # Anfangswerte für y
    if configuration_data["optimize_y0"]:
        y = np.array([element for element in individual[V_size + W_size + T_size:]])
    else:
        y = np.zeros(number_neurons)

    y = y[:, np.newaxis]

    fitness_current = 0
    ob = env.reset()
    done = False

    n = 0

    # Test fitness through simulation
    while not done:
        dy = continuous_time_rnn(y, alpha, V, W, ob)
        y = y + delta_t * dy

        # Clip to state boundaries
        y = np.clip(y, configuration_data["clipping_range_min"], configuration_data["clipping_range_max"])

        o = np.tanh(np.dot(y.T, T))
        action = o[0]

        # Perform simulation step of the environment
        ob, rew, done, info = env.step(action)

        fitness_current += rew

        env.render()
        time.sleep(0.01)

        n += 1

    print(fitness_current)
