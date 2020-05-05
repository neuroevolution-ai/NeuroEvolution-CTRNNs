import mujoco_py
import numpy as np
import time
import sys
import pickle
import gym
import json
from datetime import datetime
import os
import brains.continuous_time_rnn as ctrnn
import brains.layered_nn as lnn
import brains.feed_forward as ff

from deap import algorithms
from deap import base
from deap import creator
from deap import tools
from deap import cma
from scoop import futures


def evalFitness(individual):

    # Create brain
    brain = brain_class(input_size, output_size, individual, configuration_data)

    fitness_current = 0
    number_fitness_runs = configuration_data["number_fitness_runs"]

    for i in range(number_fitness_runs):

        if configuration_data["random_seed_for_environment"] is not -1:
            env.seed(configuration_data["random_seed_for_environment"])
        ob = env.reset()
        done = False

        while not done:

            # Perform step of the brain simulation
            action = brain.step(ob)

            # Perform step of the environment simulation
            ob, rew, done, info = env.step(action)

            fitness_current += rew

    return fitness_current/number_fitness_runs,


# Load configuration file
with open("Configuration.json", "r") as read_file:
    configuration_data = json.load(read_file)

# Get brain class
if configuration_data["neural_network_type"] == 'LNN':
    brain_class = ff.FeedForwardNN
elif configuration_data["neural_network_type"] == 'CTRNN':
    brain_class = ctrnn.ContinuousTimeRNN
else:
    sys.exit()

env = gym.make(configuration_data["environment"])

# Set random seed for gym environment
if configuration_data["random_seed_for_environment"] is not -1:
    env.seed(configuration_data["random_seed_for_environment"])

# Get individual size
input_size = env.observation_space.shape[0]
output_size = env.action_space.shape[0]

individual_size = brain_class.get_individual_size(input_size, output_size, configuration_data)

creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, typecode='b', fitness=creator.FitnessMax)

toolbox = base.Toolbox()

# Multiprocessing
toolbox.register("map", futures.map)

toolbox.register("evaluate", evalFitness)

strategy = cma.Strategy(centroid=[0.0] * individual_size, sigma=configuration_data["sigma"],
                        lambda_=configuration_data["population_size"])

toolbox.register("generate", strategy.generate, creator.Individual)
toolbox.register("update", strategy.update)

if __name__ == "__main__":

    startTime = time.time()
    startDate = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

    hof = tools.HallOfFame(5)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("std", np.std)
    stats.register("min", np.min)
    stats.register("max", np.max)

    pop, log = algorithms.eaGenerateUpdate(toolbox, ngen=configuration_data["number_generations"],
                                           stats=stats, halloffame=hof)

    # print elapsed time
    print("Time elapsed: %s" % (time.time() - startTime))

    # Create new directory to store data of current simulation run
    directory = os.path.join('Simulation_Results', startDate)
    os.makedirs(directory)

    # Save Configuration file as json
    with open(os.path.join(directory, 'Configuration.json'), 'w') as outfile:
        json.dump(configuration_data, outfile)

    # Save hall of fame individuals
    with open(os.path.join(directory, 'HallOfFame.pickle'), "wb") as fp:
        pickle.dump(hof, fp)

    # Save Log
    with open(os.path.join(directory, 'Log.json'), 'w') as outfile:
        json.dump(log, outfile)

    # Write Log to text file
    with open(os.path.join(directory, 'Log.txt'), 'w') as write_file:

        for configuration_key, configuration_value in configuration_data.items():
            write_file.write(configuration_key + ": " + str(configuration_value))
            write_file.write('\n')

        write_file.write('\n')
        write_file.write('Genome Size: {:d}\n'.format(individual_size))
        write_file.write('Inputs: {:d}\n'.format(input_size))
        write_file.write('Outputs: {:d}\n'.format(output_size))
        write_file.write('\n')
        dash = '-' * 80
        write_file.write(dash + '\n')
        write_file.write(
            '{:<8s}{:<12s}{:<16s}{:<16s}{:<16s}{:<16s}\n'.format('gen', 'nevals', 'avg', 'std', 'min', 'max'))
        write_file.write(dash + '\n')

        # Write data for each episode
        for line in log:
            write_file.write(
                '{:<8d}{:<12d}{:<16.2f}{:<16.2f}{:<16.2f}{:<16.2f}\n'.format(line['gen'], line['nevals'],
                                                                             line['avg'], line['std'], line['min'],
                                                                             line['max']))

        # Write elapsed time
        write_file.write("\nTime elapsed: %.4f seconds" % (time.time() - startTime))
