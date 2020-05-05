from brains import feed_forward as ff
from brains import layered_nn as lnn
import json
import numpy as np

with open("../Configuration.json", "r") as read_file:
    config = json.load(read_file)

input_size = 28
output_size = 8

individual = np.load("weight_data.npy")

ff_net = ff.FeedForwardNN(input_size, output_size, individual, config)

test_ob = np.load("input_data.npy")
ff_output = ff_net.step(test_ob)
np.save("output_ff", ff_output)

print("Stop")
