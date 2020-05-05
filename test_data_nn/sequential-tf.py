import tensorflow as tf
from tensorflow.keras import layers
import json
import numpy as np

input_size = 28
output_size = 8

with open("../Configuration.json", "r") as read_file:
    config = json.load(read_file)

hidden_size1 = config["number_neurons_layer1"]
hidden_size2 = config["number_neurons_layer2"]

model = tf.keras.Sequential()
model.add(layers.Dense(hidden_size1, input_dim=input_size, activation='tanh'))
model.add(layers.Dense(hidden_size2, activation='tanh'))
model.add(layers.Dense(output_size, activation='tanh'))

model.summary()

weight_data = np.load("weight_data.npy")

W1_size = input_size * hidden_size1
W2_size = hidden_size1 * hidden_size2
W3_size = hidden_size2 * output_size

index_b = W1_size + W2_size + W3_size

W1 = weight_data[:W1_size]
W2 = weight_data[W1_size:W1_size + W2_size]
W3 = weight_data[W1_size + W2_size:W1_size + W2_size + W3_size]

B1 = weight_data[index_b:index_b + hidden_size1]
B2 = weight_data[index_b + hidden_size1:index_b + hidden_size1 + hidden_size2]
B3 = weight_data[index_b+hidden_size1+hidden_size2:]

W1 = np.reshape(W1, [input_size, hidden_size1])
W2 = np.reshape(W2, [hidden_size1, hidden_size2])
W3 = np.reshape(W3, [hidden_size2, output_size])

old_weights = model.get_layer(index=0).get_weights()

model.get_layer(index=0).set_weights([W1, B1])
model.get_layer(index=1).set_weights([W2, B2])
model.get_layer(index=2).set_weights([W3, B3])

setted_weights = model.get_layer(index=0).get_weights()

assert np.array_equal(W1, setted_weights[0])
assert np.array_equal(B1, setted_weights[1])

test_ob = np.load("input_data.npy")

tensorflow_output = model.predict(test_ob[None])

ff_output = np.load("output_ff.npy")

assert np.array_equal(tensorflow_output[0], ff_output)

print("A")