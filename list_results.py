import os
import json
import argparse
import logging
import csv


def read_simulations(base_directory):
    simulation_runs = []
    for d in os.listdir(base_directory):
        simulation_folder = os.path.join(base_directory, d)

        # noinspection PyBroadException
        try:
            with open(os.path.join(simulation_folder, 'Configuration.json'), "r") as read_file:
                conf = json.load(read_file)
        except:
            conf = None
            logging.error("couldn't read conf for " + str(simulation_folder), exc_info=True)
        # noinspection PyBroadException
        try:
            with open(os.path.join(simulation_folder, 'Log.json'), 'r') as read_file:
                log = json.load(read_file)
        except Exception:
            log = None
            logging.error("couldn't read log for " + str(simulation_folder), exc_info=True)
        sim = {
            "dir": d,
            "conf": conf,
            "log": log,
        }
        simulation_runs.append(sim)
    return simulation_runs


def get_attribute_or_none(d, attr):
    if attr in d:
        return d[attr]
    return None


def gather_info_for_csv(simulation):
    log = simulation["log"]
    conf = simulation["conf"]
    generations = [i for i in range(len(log))]
    avg = [generation["avg"] for generation in log]
    maximum = [generation["max"] for generation in log]

    return {"environment": get_attribute_or_none(conf, "environment"),
            "neural_network_type": get_attribute_or_none(conf, "neural_network_type"),
            "trainer_type": get_attribute_or_none(conf, "trainer_type"),
            "number_neurons": get_attribute_or_none(conf, "number_neurons"),
            "population_size": get_attribute_or_none(conf, "population_size"),
            "parameter_perturbations": get_attribute_or_none(conf, "parameter_perturbations"),
            "gen": str(max(generations)),
            "mavg": str(max(avg)),
            "max": str(max(maximum)),
            "directory": simulation["dir"]}


logging.basicConfig()
parser = argparse.ArgumentParser(description='Visualize experiment results')
parser.add_argument('--dir', metavar='dir', type=str, help='base directory for input',
                    default='Simulation_Results/CTRNN')
parser.add_argument('--csv', metavar='type', type=str, help='location of output csv file', default='output.csv')
args = parser.parse_args()

data = []

for simulation in read_simulations(args.dir):
    data.append(gather_info_for_csv(simulation))

keys = data[0].keys()
with open(args.csv, 'w') as output_file:
    dict_writer = csv.DictWriter(output_file, keys)
    dict_writer.writeheader()
    dict_writer.writerows(data)
