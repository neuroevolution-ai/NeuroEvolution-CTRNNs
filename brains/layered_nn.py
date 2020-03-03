import torch
import numpy as np
import torch.nn as nn


class LayeredNN(nn.Module):

    def __init__(self, input_size, output_size, individual, config):

        super(LayeredNN, self).__init__()

        self.hidden_size1 = config["number_neurons_layer1"]
        self.hidden_size2 = config["number_neurons_layer2"]

        self.fc1 = nn.Linear(input_size, self.hidden_size1, bias=config["use_biases"])
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(self.hidden_size1, self.hidden_size2, bias=config["use_biases"])
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(self.hidden_size2, output_size, bias=config["use_biases"])

        self.input_size = input_size
        self.output_size = output_size

        # Get weight matrizes
        W1_size = self.input_size*self.hidden_size1
        W2_size = self.hidden_size1*self.hidden_size2
        W3_size = self.hidden_size2*self.output_size

        # Indirect encoding
        if config["indirect_encoding"]:

            config_cppn = dict()
            config_cppn["number_neurons_layer1"] = config["cppn_hidden_size1"]
            config_cppn["number_neurons_layer2"] = config["cppn_hidden_size2"]
            config_cppn["indirect_encoding"] = False
            config_cppn["use_biases"] = False

            cppn_weights = LayeredNN(4, 1, individual, config_cppn)

            self.W1 = np.zeros((self.hidden_size1, self.input_size), dtype=np.single)
            self.W2 = np.zeros((self.hidden_size2, self.hidden_size1), dtype=np.single)
            self.W3 = np.zeros((self.output_size, self.hidden_size2), dtype=np.single)

            for i, j in np.ndindex(self.W1.shape):
                self.W1[i, j] = cppn_weights.step(np.array([i/(self.hidden_size1-1), 0.33, j/(self.input_size-1),0]))

            for i, j in np.ndindex(self.W2.shape):
                self.W2[i, j] = cppn_weights.step(np.array([i/(self.hidden_size2-1),0.66, j/(self.hidden_size1-1),0.33]))

            for i, j in np.ndindex(self.W3.shape):
                self.W3[i, j] = cppn_weights.step(np.array([i/(self.output_size-1),1.0, j/(self.hidden_size2-1),0.66]))

        # Direct Encoding
        else:

            self.W1 = np.array([[float(element)] for element in individual[0:W1_size]], dtype=np.single)
            self.W2 = np.array([[float(element)] for element in individual[W1_size:W1_size+W2_size]], dtype=np.single)
            self.W3 = np.array([[float(element)] for element in individual[W1_size+W2_size:W1_size + W2_size + W3_size]],
                               dtype=np.single)

            self.W1 = self.W1.reshape([self.hidden_size1, input_size])
            self.W2 = self.W2.reshape([self.hidden_size2, self.hidden_size1])
            self.W3 = self.W3.reshape([output_size, self.hidden_size2])

            # Normalize
            # W1 = (W1 - W1.mean()) / W1.std()
            # W2 = (W2 - W2.mean()) / W2.std()
            # W3 = (W3 - W3.mean()) / W3.std()

            # Biases
            if config["use_biases"]:
                index_b = W1_size + W2_size + W3_size
                self.B1 = np.array([float(element) for element in individual[index_b:index_b + self.hidden_size1]],
                                   dtype=np.single)
                self.B2 = np.array([float(element) for element in
                                    individual[index_b + self.hidden_size1:index_b + self.hidden_size1 + self.hidden_size2]],
                                   dtype=np.single)
                self.B3 = np.array([float(element) for element in individual[index_b + self.hidden_size1 + self.hidden_size2:]],
                                   dtype=np.single)

                self.fc1.bias.data = torch.from_numpy(self.B1)
                self.fc2.bias.data = torch.from_numpy(self.B2)
                self.fc3.bias.data = torch.from_numpy(self.B3)

        self.fc1.weight.data = torch.from_numpy(self.W1)
        self.fc2.weight.data = torch.from_numpy(self.W2)
        self.fc3.weight.data = torch.from_numpy(self.W3)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu1(out)
        out = self.fc2(out)
        out = self.relu2(out)
        out = self.fc3(out)
        return out

    def step(self, ob):
        ob2 = ob[np.newaxis, :]
        ob2 = ob2.astype(np.single)
        output = self(torch.from_numpy(ob2))
        output_np = output.detach().numpy()[0, :]

        return output_np

    @staticmethod
    def get_individual_size(input_size, output_size, config):

        hidden_size1 = config["number_neurons_layer1"]
        hidden_size2 = config["number_neurons_layer2"]
        cppn_hidden_size1 = config["cppn_hidden_size1"]
        cppn_hidden_size2 = config["cppn_hidden_size2"]

        if config["indirect_encoding"]:
            individual_size = 4 * cppn_hidden_size1 + cppn_hidden_size1 * cppn_hidden_size2 + cppn_hidden_size2 * 1

            # CPPN always uses biases
            individual_size += cppn_hidden_size1 + cppn_hidden_size2 + 1

        else:
            individual_size = input_size * hidden_size1 + hidden_size1 * hidden_size2 + hidden_size2 * output_size

            if config["use_biases"]:
                individual_size += hidden_size1 + hidden_size2 + output_size

        return individual_size
