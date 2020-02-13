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
    def __init__(self, input_size, hidden_size1, hidden_size2, output_size, individual):
        super(NeuralNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size1)
        self.tanh1 = nn.Tanh()
        self.fc2 = nn.Linear(hidden_size1, hidden_size2)
        self.tanh2 = nn.Tanh()
        self.fc3 = nn.Linear(hidden_size2, output_size)
        self.tanh3 = nn.Tanh()

        self.W1, self.W2, self.W3 = self._get_weight_matrizes(individual)
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

    def _get_weight_matrizes(self, individual):

        W1_size = input_size*hidden_size1
        W2_size = hidden_size1*hidden_size2
        # W3_size = hidden_size2*output_size

        W1 = np.array([[float(element)] for element in individual[0:W1_size]], dtype=np.single)
        W2 = np.array([[float(element)] for element in individual[W1_size:W1_size+W2_size]], dtype=np.single)
        W3 = np.array([[float(element)] for element in individual[W1_size+W2_size:]], dtype=np.single)

        W1 = W1.reshape([hidden_size1, input_size])
        W2 = W2.reshape([hidden_size2, hidden_size1])
        W3 = W3.reshape([output_size, hidden_size2])

        # Normalize
        # W1 = (W1 - W1.mean()) / W1.std()
        W2 = (W2 - W2.mean()) / W2.std()
        # W3 = (W3 - W3.mean()) / W3.std()

        return W1, W2, W3

    def get_weight_matrizes(self):
        return self.W1, self.W2, self.W3


def evalFitness(individual):

    # model = NeuralNet(input_size, hidden_size1, hidden_size2, output_size, individual)

    fitness_current = 0
    ob = env.reset()
    done = False

    # This does not work with multiprocessing
    # global number_steps

    # Test fitness through simulation
    while not done:
        # action = model.get_action(ob)
        action = env.action_space.sample()
        ob, rew, done, info = env.step(action)
        fitness_current += rew

    return fitness_current,


# env = gym.make("AntBulletEnv-v0")
# env = gym.make("HalfCheetahBulletEnv-v0")
# env = gym.make("CartPoleBulletEnv-v1")
# env = gym.make("Walker2DBulletEnv-v0")
# env = gym.make("InvertedPendulumSwingupBulletEnv-v0")
env = gym.make('Humanoid-v2')


# Hyper-parameters
input_size = env.observation_space.shape[0]
hidden_size1 = 10
hidden_size2 = 5
output_size = env.action_space.shape[0]

# Size of Individual
IND_SIZE=input_size*hidden_size1+hidden_size1*hidden_size2+hidden_size2*output_size

creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, typecode='b', fitness=creator.FitnessMax)

toolbox = base.Toolbox()

# Multiprocessing
toolbox.register("map", futures.map)
toolbox.register("evaluate", evalFitness)

print(IND_SIZE)

# strategy = cma.Strategy(centroid=[0.0] * IND_SIZE, sigma=5.0, {'bounds': [0, np.inf]})
# strategy = cma.Strategy(centroid=[0.0] * IND_SIZE, sigma=5.0, lambda_=100)
strategy = cma.Strategy(centroid=[0.0] * IND_SIZE, sigma=5.0)
toolbox.register("generate", strategy.generate, creator.Individual)
toolbox.register("update", strategy.update)

number_steps = 0

if __name__ == "__main__":
    random.seed(64)

    startTime = time.time()

    hof = tools.HallOfFame(5)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("std", np.std)
    stats.register("min", np.min)
    stats.register("max", np.max)

    pop, log = algorithms.eaGenerateUpdate(toolbox, ngen=100, stats=stats, halloffame=hof)
