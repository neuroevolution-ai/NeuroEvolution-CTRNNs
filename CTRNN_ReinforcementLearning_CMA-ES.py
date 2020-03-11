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

from deap import tools
from scoop import futures

from Others.core import EpisodeRunner
from Others.trainer_CMA_ES import TrainerCmaEs


# Load configuration file
with open("Configuration.json", "r") as read_file:
    configuration_data = json.load(read_file)

# Get brain class
if configuration_data["neural_network_type"] == 'LNN':
    brain_class = lnn.LayeredNN
elif configuration_data["neural_network_type"] == 'CTRNN':
    brain_class = ctrnn.ContinuousTimeRNN
else:
    raise RuntimeError("unknown neural_network_type: " + str(configuration_data["neural_network_type"]))

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
    discrete_actions = True

individual_size = brain_class.get_individual_size(input_size, output_size, configuration_data)

ep_runner = EpisodeRunner(conf=configuration_data, discrete_actions=discrete_actions, brain_class=brain_class,
                          input_size=input_size, output_size=output_size, env=env)

if configuration_data["trainer_type"] == "CMA_ES":
    trainer = TrainerCmaEs(map_func=futures.map, individual_size=individual_size,
                           evalFitness=ep_runner.evalFitness, conf=configuration_data)
else:
    raise RuntimeError("unknown trainer_type: " + str(configuration_data["trainer_type"]))

if __name__ == "__main__":

    startTime = time.time()
    startDate = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("std", np.std)
    stats.register("min", np.min)
    stats.register("max", np.max)

    pop, log = trainer.train(stats)

    # print elapsed time
    print("Time elapsed: %s" % (time.time() - startTime))

    # Create new directory to store data of current simulation run
    subdirectory_name = configuration_data["neural_network_type"]
    directory = os.path.join('Simulation_Results', subdirectory_name, startDate)
    os.makedirs(directory)
    print("output directory: " + str(directory))

    # Save Configuration file as json
    with open(os.path.join(directory, 'Configuration.json'), 'w') as outfile:
        json.dump(configuration_data, outfile)

    # Save hall of fame individuals
    with open(os.path.join(directory, 'HallOfFame.pickle'), "wb") as fp:
        pickle.dump(trainer.hof, fp)

    # Save Log
    with open(os.path.join(directory, 'Log.json'), 'w') as outfile:
        json.dump(log, outfile)

    # Write Log to text file
    with open(os.path.join(directory, 'Log.txt'), 'w') as write_file:

        write_file.write('Gym environment: {:s}\n'.format(configuration_data["environment"]))
        write_file.write('Random seed for environment: {:d}\n'.format(configuration_data["random_seed_for_environment"]))
        write_file.write('Number of neurons: {:d}\n'.format(configuration_data["number_neurons"]))
        write_file.write('Time step (delta_t): {:.3f}\n'.format(configuration_data["delta_t"]))
        write_file.write('Optimize state boundaries: {}\n'.format(configuration_data["optimize_state_boundaries"]))
        write_file.write('Clipping range max: {:.2f}\n'.format(configuration_data["clipping_range_max"]))
        write_file.write('Clipping range min: {:.2f}\n'.format(configuration_data["clipping_range_min"]))
        write_file.write('Optimize initial states y0: {}\n'.format(configuration_data["optimize_y0"]))
        write_file.write('Set principal elements of W to negative: {}\n'.format(configuration_data["set_principle_diagonal_elements_of_W_negative"]))
        write_file.write('Population size: {:d}\n'.format(configuration_data["population_size"]))
        write_file.write('Number of Generations: {:d}\n'.format(configuration_data["number_generations"]))
        write_file.write('Sigma: {:.2f}\n'.format(configuration_data["sigma"]))
        write_file.write('Number of runs per evaluation: {:d}\n'.format(configuration_data["number_fitness_runs"]))
        write_file.write('\n')
        write_file.write('Genome Size: {:d}\n'.format(individual_size))
        write_file.write('Inputs: {:d}\n'.format(input_size))
        write_file.write('Outputs: {:d}\n'.format(output_size))
        write_file.write('\n')
        write_file.write('Number of neurons in hidden layer 1: {:d}\n'.format(configuration_data["number_neurons_layer1"]))
        write_file.write('Number of neurons in hidden layer 2: {:d}\n'.format(configuration_data["number_neurons_layer2"]))
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
