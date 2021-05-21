import subprocess
import json
import random
import sys

optimization_configuration_keys = ["experiment_id",
                                   "environment",
                                   "neural_network_type",
                                   "random_seed_for_environment",
                                   "population_size",
                                   "number_generations",
                                   "sigma",
                                   "number_fitness_runs"]

lnn_configuration_keys = ["use_biases", "number_neurons_layer1"]

ctrnn_configuration_keys = ["number_neurons",
                            "delta_t",
                            "optimize_state_boundaries",
                            "clipping_range",
                            "optimize_y0",
                            "set_principle_diagonal_elements_of_W_negative"]


with open('Stop_Optimization.json', 'w') as outfile:
    json.dump({"stop_optimization": False}, outfile)

stop_optimization = False

print("Optimization started")

while not stop_optimization:
    
    with open("Configuration_DesignSpace.json", "r") as readfile:
        configuration = json.load(readfile)

    configuration_temp = dict()

    for key, value in configuration.items():
        if isinstance(value, list):
            configuration_temp[key] = random.choice(value)
        else:
            configuration_temp[key] = value

    # Get neural network type
    configuration_keys = []
    configuration_keys.extend(optimization_configuration_keys)
    if configuration_temp["neural_network_type"] == "LNN":
        configuration_keys.extend(lnn_configuration_keys)
    elif configuration_temp["neural_network_type"] == "CTRNN":
        configuration_keys.extend(ctrnn_configuration_keys)
    else:
        print("Neural network type not defined")
        sys.exit()

    # Remove not required keys from current configuration
    configuration_out = dict()
    for key in configuration_keys:
        configuration_out[key] = configuration_temp[key]

    if "clipping_range" in configuration_out and "optimize_state_boundaries" in configuration_out:
        if configuration_out["optimize_state_boundaries"]:
            del configuration_out['clipping_range']

    if "number_neurons_layer1" in configuration_out:
        configuration_out["number_neurons_layer2"] = round(configuration_out["number_neurons_layer1"] / 2)

    with open('Configuration.json', 'w') as outfile:
        json.dump(configuration_out, outfile)

    with open("Stop_Optimization.json", "r") as readfile:
        d = json.load(readfile)
        stop_optimization = d["stop_optimization"]

    subprocess.run(["python", "-m", "scoop", "CTRNN_ReinforcementLearning_CMA-ES.py"])

print("Optimization finished")
