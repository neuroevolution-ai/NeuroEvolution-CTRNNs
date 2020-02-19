import mujoco_py
import gym
import pickle

import numpy as np
import time

def model1_np(y, alpha, V, W, u):

    u2 = u[:, np.newaxis]

    # Differential equation
    dydt = np.dot(W, np.tanh(y)) + np.dot(V, u2)

    return dydt


# Load halloffame candidate
with open("halloffame_individual1.pickle", "rb") as fp:
    individual = pickle.load(fp)

env = gym.make('Walker2d-v2')
# env.render()
env.reset()

# test = gym.envs.registry.all()
fitness_current = 0
done = False

# Number of neurons
number_neurons = 30
input_size = env.observation_space.shape[0]
output_size = env.action_space.shape[0]

# Alpha
alpha = 0.01

delta_t = 0.05

# Size of Individual
IND_SIZE = input_size * number_neurons + number_neurons * number_neurons + number_neurons * output_size

number_steps = 0

V_size = input_size * number_neurons
W_size = number_neurons * number_neurons

# Get weight matrizes of current individual
V = np.array([[element] for element in individual[0:V_size]])
W = np.array([[element] for element in individual[V_size:V_size + W_size]])
T = np.array([[element] for element in individual[V_size + W_size:]])

V = V.reshape([number_neurons, input_size])
W = W.reshape([number_neurons, number_neurons])
T = T.reshape([number_neurons, output_size])

# Set elements of main diagonal to less than 0
for j in range(number_neurons):
    W[j][j] = -abs(W[j][j])

for i in range(1):

    # Anfangswerte
    y = np.zeros(number_neurons)
    y = y[:, np.newaxis]

    fitness_current = 0
    ob = env.reset()
    done = False

    n = 0
    y_list = []
    # Test fitness through simulation
    while not done:
        dy = model1_np(y, alpha, V, W, ob)
        y = y + delta_t * dy

        y = np.clip(y, -1.0, 1.0)

        o = np.dot(y.T, T)

        o2 = np.tanh(o)

        action = o2[0]
        ob, rew, done, info = env.step(action)
        fitness_current += rew

        env.render()
        #print(y.T)
        #print(action)
        time.sleep(0.01)

        y_list.append(y.tolist())
        n += 1

    print(fitness_current)
