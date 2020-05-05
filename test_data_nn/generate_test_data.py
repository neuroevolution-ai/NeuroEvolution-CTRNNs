from brains import feed_forward as ff
from brains import layered_nn as lnn
import json
import numpy as np

with open("../Configuration.json", "r") as read_file:
    config = json.load(read_file)

input_size = 28
output_size = 8

individual_size = ff.FeedForwardNN.get_individual_size(input_size, output_size, config)

weights = np.random.rand(individual_size).astype(np.single)

np.save("weight_data", weights)

input_data = np.random.rand(input_size).astype(np.single)

np.save("input_data", input_data)