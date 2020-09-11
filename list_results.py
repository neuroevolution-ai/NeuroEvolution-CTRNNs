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
        except Exception:
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


def gather_info_for_csv(simulation, conf_keys):
    log = simulation["log"]
    conf = simulation["conf"]
    avg = [generation["avg"] for generation in log]
    maximum = [generation["max"] for generation in log]

    info = dict()

    for key in conf_keys:
        info[key.replace('_',' ')] = get_attribute_or_none(conf, key)

    info["maximum average"] = str(max(avg))
    info["maximum"] = str(max(maximum))
    info["directory"] = simulation["dir"]

    return info


logging.basicConfig()
parser = argparse.ArgumentParser(description='Visualize experiment results')
parser.add_argument('--dir', metavar='dir', type=str, help='base directory for input',
                    default='Simulation_Results')
parser.add_argument('--csv', metavar='type', type=str, help='location of output csv file', default='output.csv')
args = parser.parse_args()

configuration_keys = []
for simulation in read_simulations(args.dir):
    for key in simulation["conf"].keys():
        if key not in configuration_keys:
            configuration_keys.append(key)

configuration_keys.remove("delta_t")

header = [key.replace('_',' ') for key in configuration_keys]
header.extend(["maximum average", "maximum", "directory"])

data = []
for simulation in read_simulations(args.dir):
    data.append(gather_info_for_csv(simulation, configuration_keys))

with open(args.csv, 'w') as output_file:
    dict_writer = csv.DictWriter(output_file, header)
    dict_writer.writeheader()
    dict_writer.writerows(data)
