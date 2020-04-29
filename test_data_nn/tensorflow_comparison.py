import tensorflow as tf
import json
import numpy as np

tf.enable_eager_execution()
input_size = 28
output_size = 8

with open("../Configuration.json", "r") as read_file:
    config = json.load(read_file)

hidden_size1 = config["number_neurons_layer1"]
hidden_size2 = config["number_neurons_layer2"]

W1_size = input_size * hidden_size1
W2_size = hidden_size1 * hidden_size2
W3_size = hidden_size2 * output_size

individual = np.load("weight_data.npy")

input_layer = x = tf.keras.Input((28,))
layer_1 = tf.keras.layers.Dense(hidden_size1, activation=tf.keras.activations.relu)
layer_2 = tf.keras.layers.Dense(hidden_size2, activation=tf.keras.activations.relu)
output_layer = tf.keras.layers.Dense(output_size, activation=tf.keras.activations.relu)

x = layer_1(x)
x = layer_2(x)
a = output_layer(x)

model = tf.keras.Model(inputs=input_layer, outputs=a)
model.summary()

W1 = np.array([[float(element)] for element in individual[:W1_size]])
W2 = np.array([[float(element)] for element in individual[W1_size:W1_size+W2_size]])
W3 = np.array([[float(element)] for element in individual[W1_size+W2_size:W1_size+W2_size+W3_size]])

index_b = W1_size + W2_size + W3_size

B1 = np.array([[float(element)] for element in individual[index_b:index_b + hidden_size1]])
B2 = np.array([[float(element)] for element in individual[index_b + hidden_size1:index_b + hidden_size1 + hidden_size2]])
B3 = np.array([[float(element)] for element in individual[index_b + hidden_size1 + hidden_size2:]])

W1 = W1.reshape([hidden_size1, input_size])
#W1_ = np.transpose(W1)
W1 = np.transpose(W1)
W2 = W2.reshape([hidden_size2, hidden_size1])
W2 = np.transpose(W2)
W3 = W3.reshape([output_size, hidden_size2])
W3 = np.transpose(W3)

fnn_w1 = np.load("ff_w1.npy")
fnn_w2 = np.load("ff_w2.npy")
fnn_w3 = np.load("ff_w3.npy")
fnn_b1 = np.load("ff_b1.npy")
fnn_b2 = np.load("ff_b2.npy")
fnn_b3 = np.load("ff_b3.npy")




layer_1.set_weights([W1, B1])
layer_2.set_weights([W2, B2])
output_layer.set_weights([W3, B3])


ob_test = np.load("input_data.npy")[None]

tf_output = model.predict(ob_test)

fnn_ouput = np.load("output_ff.npy")

assert np.array_equal(tf_output, fnn_ouput)

print("s")