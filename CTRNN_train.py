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

from Others.trainer_CMA_ES import TrainerCmaEs
from Others.trainer_mu_plus_lambda import TrainerMuPlusLambda
from Others.result_handler import ResultHandler
import argparse


class EpisodeRunner(object):
    # episode Runner can't be defined in a submodule, because
    # when it is, scoop throws lots of errors on process-end.

    def __init__(self, conf, discrete_actions, brain_class, input_size, output_size, env):
        self.conf = conf
        self.discrete_actions = discrete_actions
        self.brain_class = brain_class
        self.input_size = input_size
        self.output_size = output_size
        self.env = env

    def evalFitness(self, individual):
        brain = self.brain_class(self.input_size, self.output_size, individual, self.conf)
        fitness_current = 0
        number_fitness_runs = self.conf["number_fitness_runs"]

        for i in range(number_fitness_runs):
            ob = self.env.reset()
            done = False
            consecutive_non_movement = 0
            while not done:
                # Perform step of the brain simulation
                action = brain.step(ob)
                if self.discrete_actions:
                    action = np.argmax(action)
                # Perform step of the environment simulation
                ob, rew, done, info = self.env.step(action)
                if self.conf["environment"] == "BipedalWalker-v3":
                    if ob[2] < 0.0001:
                        consecutive_non_movement = consecutive_non_movement + 1
                        if consecutive_non_movement > 50:
                            done = True
                            rew = rew - 100
                    else:
                        consecutive_non_movement = 0
                fitness_current += rew
        return fitness_current / number_fitness_runs,


parser = argparse.ArgumentParser(description='train CTRNN')
parser.add_argument('--from-checkpoint', metavar='dir', type=str,
                    help='continues training from a checkpoint', default=None)
parser.add_argument('--configuration', metavar='dir', type=str,
                    help='use an alternative configuration file', default='Configuration.json')
args = parser.parse_args()


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
                           evalFitness=ep_runner.evalFitness, conf=configuration_data,
                           )
elif configuration_data["trainer_type"] == "MU_LAMBDA":
    trainer = TrainerMuPlusLambda(map_func=futures.map, individual_size=individual_size,
                                  evalFitness=ep_runner.evalFitness,
                                  population_size=configuration_data["population_size"],
                                  trainer_parameters=configuration_data["MU_LAMBDA_parameters"], )
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
    result_handler = ResultHandler(startDate=startDate,
                                   nn_type=configuration_data["neural_network_type"],
                                   configuration_data=configuration_data)
    pop, log = trainer.train(stats, number_generations=configuration_data["number_generations"],
                             checkpoint=args.from_checkpoint)

    # print elapsed time
    print("Time elapsed: %s" % (time.time() - startTime))
    result_handler.write_result(
        hof=trainer.hof,
        log=log,
        time_elapsed=(time.time() - startTime),
        output_size=output_size,
        input_size=input_size,
        individual_size=individual_size)
    print("done")