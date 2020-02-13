import random
import gym
import torch
import numpy as np
import torch.nn as nn
import time
import pickle
import matplotlib.pyplot as plt

from deap import algorithms
from deap import base
from deap import creator
from deap import tools
from deap import cma
from scoop import futures

# https://github.com/hardmaru/estool/blob/master/config.py

# Fully connected neural network with one hidden layer
class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size1, hidden_size2, output_size, individual, indirect_encoding=False):
        super(NeuralNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size1)
        self.tanh1 = nn.Tanh()
        self.fc2 = nn.Linear(hidden_size1, hidden_size2)
        self.tanh2 = nn.Tanh()
        self.fc3 = nn.Linear(hidden_size2, output_size)
        self.tanh3 = nn.Tanh()

        self.input_size = input_size
        self.hidden_size1 = hidden_size1
        self.hidden_size2 = hidden_size2
        self.output_size = output_size

        self.W1, self.W2, self.W3 = self._set_weight_matrizes(individual, indirect_encoding)
        self.fc1.weight.data = torch.from_numpy(self.W1)
        self.fc2.weight.data = torch.from_numpy(self.W2)
        self.fc3.weight.data = torch.from_numpy(self.W3)

    def forward(self, x):
        out = self.fc1(x)
        out = self.tanh1(out)
        out = self.fc2(out)
        out = self.tanh2(out)
        out = self.fc3(out)
        out = self.tanh3(out)
        return out

    def get_action(self, ob):
        ob2 = ob[np.newaxis, :]
        ob2 = ob2.astype(np.single)
        output = self(torch.from_numpy(ob2))
        output_np = output.detach().numpy()[0, :]

        return output_np

    def _set_weight_matrizes(self, individual, indirect_encoding):

        W1_size = self.input_size*self.hidden_size1
        W2_size = self.hidden_size1*self.hidden_size2
        # W3_size = hidden_size2*output_size

        # indirect encoding
        if indirect_encoding:
            cppn = NeuralNet(4, cppn_hidden_size1, cppn_hidden_size2, 1, individual, indirect_encoding=False)

            W1 = np.zeros((self.hidden_size1, self.input_size), dtype=np.single)
            W2 = np.zeros((self.hidden_size2, self.hidden_size1), dtype=np.single)
            W3 = np.zeros((self.output_size, self.hidden_size2), dtype=np.single)

            for i, j in np.ndindex(W1.shape):
                temp = np.array([i/(self.hidden_size1-1),0.33,j/(self.input_size-1),0])
                W1[i, j] = cppn.get_action(temp)

            for i, j in np.ndindex(W2.shape):
                W2[i, j] = cppn.get_action(np.array([i/(self.hidden_size2-1),0.66,j/(self.hidden_size1-1),0.33]))

            for i, j in np.ndindex(W3.shape):
                W3[i, j] = cppn.get_action(np.array([i/(self.output_size-1),1.0,j/(self.hidden_size2-1),0.66]))

        else:
            W1 = np.array([[float(element)] for element in individual[0:W1_size]], dtype=np.single)
            W2 = np.array([[float(element)] for element in individual[W1_size:W1_size+W2_size]], dtype=np.single)
            W3 = np.array([[float(element)] for element in individual[W1_size+W2_size:]], dtype=np.single)

            W1 = W1.reshape([self.hidden_size1, self.input_size])
            W2 = W2.reshape([self.hidden_size2, self.hidden_size1])
            W3 = W3.reshape([self.output_size, self.hidden_size2])

        # Normalize
        W1 = (W1 - W1.mean()) / max(W1.std(), 0.1)
        W2 = (W2 - W2.mean()) / max(W2.std(), 0.1)
        W3 = (W3 - W3.mean()) / max(W3.std(), 0.1)

        return W1, W2, W3

    def get_weight_matrizes(self):
        return self.W1, self.W2, self.W3


def evalFitness(individual):

    model = NeuralNet(input_size, hidden_size1, hidden_size2, output_size, individual, indirect_encoding=True)

    fitness_current = 0
    ob = env.reset()
    done = False

    # This does not work with multiprocessing
    global number_steps

    # Test fitness through simulation
    while not done:
        action = model.get_action(ob)
        ob, rew, done, info = env.step(action)
        number_steps = number_steps + 1
        fitness_current += rew

    return fitness_current,


# env = gym.make("AntBulletEnv-v0")
# env = gym.make("HalfCheetahBulletEnv-v0")
# env = gym.make("CartPoleBulletEnv-v1")
# env = gym.make("Walker2DBulletEnv-v0")
# env = gym.make("InvertedPendulumSwingupBulletEnv-v0")
env = gym.make('HalfCheetah-v2')

# Hyper-parameters
input_size = env.observation_space.shape[0]
hidden_size1 = 32
hidden_size2 = 8
output_size = env.action_space.shape[0]

cppn_hidden_size1 = 12
cppn_hidden_size2 = 6

# Size of Individual
IND_SIZE=input_size*hidden_size1+hidden_size1*hidden_size2+hidden_size2*output_size
#IND_SIZE=4*cppn_hidden_size1+cppn_hidden_size1*cppn_hidden_size2+cppn_hidden_size2*1

print(IND_SIZE)

creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, typecode='b', fitness=creator.FitnessMax)

toolbox = base.Toolbox()

# Multiprocessing
toolbox.register("map", futures.map)
toolbox.register("evaluate", evalFitness)

strategy = cma.Strategy(centroid=[0.0] * IND_SIZE, sigma=1.0, lambda_= 200)
# strategy = cma.Strategy(centroid=[0.0] * IND_SIZE, sigma=5.0)
toolbox.register("generate", strategy.generate, creator.Individual)
toolbox.register("update", strategy.update)

number_steps = 0

if __name__ == "__main__":

    startTime = time.time()

    hof = tools.HallOfFame(5)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("std", np.std)
    stats.register("min", np.min)
    stats.register("max", np.max)

    pop, log = algorithms.eaGenerateUpdate(toolbox, ngen=2000, stats=stats, halloffame=hof)

    best_individual = hof[0]

    best_model = NeuralNet(input_size, hidden_size1, hidden_size2, output_size, best_individual, indirect_encoding=False)
    W1, W2, W3 = best_model.get_weight_matrizes()

    # Save weights of hof individual
    with open("Weights_hof.pickle", "wb") as fp:
        pickle.dump([W1, W2, W3], fp)

    # Get statistics from log
    generations = [i for i in range(len(log))]
    avg = [generation["avg"] for generation in log]
    high = [generation["avg"] + generation["std"] for generation in log]
    low = [generation["avg"] - generation["std"] for generation in log]

    # print results
    print("Time elapsed: %s" % (time.time() - startTime))
    print("Number of steps: %s" % number_steps)

    # Plot results
    plt.plot(generations, avg, 'r-')
    plt.plot(generations, high, 'y--')
    plt.plot(generations, low, 'b--')
    plt.xlabel('Generations')
    plt.legend(['avg', 'std high', 'std low'])
    plt.grid()
    plt.show()
