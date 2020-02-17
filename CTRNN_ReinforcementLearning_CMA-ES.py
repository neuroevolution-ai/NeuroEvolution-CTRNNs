import mujoco_py
import random
import numpy as np
import time
import pickle
import matplotlib.pyplot as plt
import gym, pybullet_envs

from deap import algorithms
from deap import base
from deap import creator
from deap import tools
from deap import cma
from scoop import futures


def model1_np(y, alpha, V, W, u):

    u2 = u[:, np.newaxis]

    # Differential equation
    dydt = np.dot(W, np.tanh(y)) + np.dot(V, u2)

    return dydt


def evalFitness(individual):

    V_size = input_size * number_neurons
    W_size = number_neurons * number_neurons

    # Get weight matrizes of current individual
    V = np.array([[element] for element in individual[0:V_size]])
    W = np.array([[element] for element in individual[V_size:V_size+W_size]])
    T = np.array([[element] for element in individual[V_size+W_size:]])

    V = V.reshape([number_neurons, input_size])
    W = W.reshape([number_neurons, number_neurons])
    T = T.reshape([number_neurons, output_size])

    # Set elements of main diagonal to less than 0
    for j in range(number_neurons):
        W[j][j] = -abs(W[j][j])

    fitness_current = 0

    for i in range(1):

        # Anfangswerte
        y = np.zeros(number_neurons)
        y = y[:, np.newaxis]

        ob = env.reset()
        done = False

        # Test fitness through simulation
        #n = 0
        while not done:
            dy = model1_np(y, alpha, V, W, ob)
            y = y + delta_t * dy
            o = np.dot(y.T, T)

            o2 = np.tanh(o)

            action = o2[0]
            action2 = env.action_space.sample()
            ob, rew, done, info = env.step(action)

            fitness_current += rew

            #n += 1
            #if n > 200:
            #    fitness_current += rew
            #if n == 400:
            #    break

    return fitness_current,


# env = gym.make('MountainCarContinuous-v0')
# env = gym.make('Swimmer-v2')
# env = gym.make('Hopper-v2')
# env = gym.make('Ant-v2')
env = gym.make('Walker2d-v2')

# Number of neurons
number_neurons = 30
input_size = env.observation_space.shape[0]
output_size = env.action_space.shape[0]

print(input_size)
print(output_size)

# Alpha
alpha = 0.01

delta_t = 0.05

# Size of Individual
IND_SIZE = input_size * number_neurons + number_neurons * number_neurons + number_neurons * output_size

print(IND_SIZE)

creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, typecode='b', fitness=creator.FitnessMax)

toolbox = base.Toolbox()

# Multiprocessing
toolbox.register("map", futures.map)

toolbox.register("evaluate", evalFitness)
strategy = cma.Strategy(centroid=[0.0] * IND_SIZE, sigma=1.0, lambda_= 200)
# strategy = cma.Strategy(centroid=[0.0] * IND_SIZE, sigma=1.0)
toolbox.register("generate", strategy.generate, creator.Individual)
toolbox.register("update", strategy.update)

if __name__ == "__main__":

    startTime = time.time()

    hof = tools.HallOfFame(5)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("std", np.std)
    stats.register("min", np.min)
    stats.register("max", np.max)

    pop, log = algorithms.eaGenerateUpdate(toolbox, ngen=1500, stats=stats, halloffame=hof)

    for i in range(len(hof)):

        # Save weights of hof individual
        with open("halloffame_individual" + str(i) + ".pickle", "wb") as fp:
            pickle.dump(list(hof[i]), fp)

    # Get statistics from log
    generations = [i for i in range(len(log))]
    avg = [generation["avg"] for generation in log]
    high = [generation["avg"] + generation["std"] for generation in log]
    low = [generation["avg"] - generation["std"] for generation in log]

    # print results
    print("Time elapsed: %s" % (time.time() - startTime))

    # Plot results
    plt.plot(generations, avg, 'r-')
    plt.plot(generations, high, 'y--')
    plt.plot(generations, low, 'b--')
    plt.xlabel('Generations')
    plt.legend(['avg', 'std high', 'std low'])
    plt.grid()
    plt.show()
