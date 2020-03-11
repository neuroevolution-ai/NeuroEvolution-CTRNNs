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
from Others.trainer_mu_plus_lambda import TrainerMuPlusLambda
from Others.result_handler import ResultHandler

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
elif configuration_data["trainer_type"] == "MU_LAMBDA":
    trainer = TrainerMuPlusLambda(map_func=futures.map, individual_size=individual_size,
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
    result_handler = ResultHandler(startDate=startDate,
                                   nn_type=configuration_data["neural_network_type"],
                                   configuration_data=configuration_data)
    pop, log = trainer.train(stats)

    # print elapsed time
    print("Time elapsed: %s" % (time.time() - startTime))
    result_handler.write_result(
        hof=trainer.hof,
        log=log,
        time_elapsed=(time.time() - startTime),
        output_size=output_size,
        input_size=input_size,
        individual_size=individual_size)
