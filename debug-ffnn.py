from brains import feed_forward as ff
from brains import layered_nn as lnn
import json
import numpy as np

with open("Configuration.json", "r") as read_file:
    config = json.load(read_file)

input_size = 28
output_size = 8

# ff_results = []
# lnn_results = []
# for i in range(10):
#     individual = np.load("test_data_nn/weight_data.npy")
#
#     ff_net = ff.FeedForwardNN(input_size, output_size, individual, config)
#     lnn_net = lnn.LayeredNN(input_size, output_size, individual, config)
#
#     test_ob = np.load("test_data_nn/input_data.npy")
#
#     ff_output = ff_net.step(test_ob)
#     lnn_output = lnn_net.step(test_ob)
#
#     ff_results.append(ff_output)
#     lnn_results.append((lnn_output))
#
#     if i == 0:
#         np.save("test_data_nn/output_ff", ff_output)

individual = np.load("test_data_nn/weight_data.npy")

ff_net = ff.FeedForwardNN(input_size, output_size, individual, config)

np.save("ff_w1", ff_net.W1)
np.save("ff_w2", ff_net.W2)
np.save("ff_w3", ff_net
np.save("ff_b1", ff_net.B1)
np.save("ff_b2", ff_net.B2)
np.save("ff_b3", ff_net.B3)

#assert np.array_equal(ff_output, lnn_output)
print("Stop")

# w_1_arr = []
# w_2_arr = []
# w_3_arr = []
#
# b_1_arr = []
# b_2_arr = []
# b_3_arr = []
# for _ in range(10):
#     individual = np.load("test_data_nn/input_data.npy")
#
#     ff_net = ff.FeedForwardNN(input_size, output_size, individual, config)
#
#     w_1_arr.append(ff_net.W1)
#     w_2_arr.append(ff_net.W2)
#     w_3_arr.append(ff_net.W3)
#     b_1_arr.append(ff_net.B1)
#     b_2_arr.append(ff_net.B2)
#     b_3_arr.append(ff_net.B3)
#
#
# for x in w_1_arr:
#     for y in w_1_arr:
#         assert np.array_equal(x, y)
#
# for x in w_2_arr:
#     for y in w_2_arr:
#         assert np.array_equal(x, y)
#
# for x in w_3_arr:
#     for y in w_3_arr:
#         assert np.array_equal(x, y)
#
# for x in b_1_arr:
#     for y in b_1_arr:
#         assert np.array_equal(x, y)
#
# for x in b_2_arr:
#     for y in b_2_arr:
#         assert np.array_equal(x, y)
#
# for x in b_3_arr:
#     for y in b_3_arr:
#         assert np.array_equal(x, y)


