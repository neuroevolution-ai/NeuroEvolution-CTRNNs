from brains import feed_forward as ff
from brains import layered_nn as lnn
import json
import numpy as np
import time
import torch

with open("Configuration.json", "r") as read_file:
    config = json.load(read_file)

input_size = 28
output_size = 8

individual_size = ff.FeedForwardNN.get_individual_size(input_size, output_size, config)
individual = np.random.uniform(-1, 1, individual_size).astype(np.float64)

inputs = []
for i in range(10000):
    inputs.append(np.random.uniform(-1, 1, input_size).astype(np.float64))

numpy_net = ff.FeedForwardNN(input_size, output_size, individual, config)
pytorch_net = lnn.LayeredNN(input_size, output_size, individual, config)

numpy_outputs = []
pytorch_outputs = []

numpy_times = []
pytorch_times = []

no_grad = True

for input in inputs:
    time_s_numpy = time.time()
    numpy_outputs.append(numpy_net.step(input))
    numpy_times.append(time.time() - time_s_numpy)

    if no_grad:
        with torch.no_grad():
            time_s_pytorch = time.time()
            pytorch_outputs.append(pytorch_net.step(input))
            pytorch_times.append(time.time() - time_s_pytorch)
    else:
        time_s_pytorch = time.time()
        pytorch_outputs.append(pytorch_net.step(input))
        pytorch_times.append(time.time() - time_s_pytorch)

all_close_counter = 0
for i in range(len(numpy_outputs)):
    all_close = np.allclose(numpy_outputs[i], pytorch_outputs[i], rtol=1e-4)
    if not all_close:
        all_close_counter += 1

print("All close false percentage: {}% from {}".format((all_close_counter/len(numpy_outputs))*100, len(numpy_outputs)))

numpy_time_mean = np.mean(numpy_times)
numpy_time_std = np.std(numpy_times)
numpy_time_max = np.max(numpy_times)
numpy_time_min = np.min(numpy_times)

pytorch_time_mean = np.mean(pytorch_times)
pytorch_time_std = np.std(pytorch_times)
pytorch_time_max = np.max(pytorch_times)
pytorch_time_min = np.min(pytorch_times)


print("NumPy Mean {} | PyTorch Mean {}".format(numpy_time_mean, pytorch_time_mean))
print("Ratio PyTorch Mean to NumPy Mean: {}".format(pytorch_time_mean/numpy_time_mean))