import subprocess
import json
import random

with open('Stop_Optimization.json', 'w') as outfile:
    json.dump({"stop_optimization": False}, outfile)

stop_optimization = False

while not stop_optimization:
    
    with open("Configuration_DesignSpace.json", "r") as readfile:
        configuration = json.load(readfile)

    configuration_new = dict()

    for key, value in configuration.items():
        if isinstance(value, list):
            configuration_new[key] = random.choice(value)
        else:
            configuration_new[key] = value

    if configuration_new["optimize_state_boundaries"] and 'clipping_range' in configuration_new:
        del configuration_new['clipping_range']

    with open('Configuration.json', 'w') as outfile:
        json.dump(configuration_new, outfile)

    with open("Stop_Optimization.json", "r") as readfile:
        d = json.load(readfile)
        stop_optimization = d["stop_optimization"]

    subprocess.run(["python", "-m", "scoop", "CTRNN_ReinforcementLearning_CMA-ES.py"])

print("Optimization finished")
