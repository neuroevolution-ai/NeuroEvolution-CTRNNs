import torch
import numpy as np
import torch.nn as nn


class LayeredNN(nn.Module):

    def __init__(self, input_size, output_size, individual, config):

        super(LayeredNN, self).__init__()

        hidden_size1 = config["number_neurons_layer1"]
        hidden_size2 = config["number_neurons_layer2"]

        self.fc1 = nn.Linear(input_size, hidden_size1)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size1, hidden_size2)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(hidden_size2, output_size)

        self.input_size = input_size
        self.hidden_size1 = hidden_size1
        self.hidden_size2 = hidden_size2
        self.output_size = output_size

        # Get weight matrizes
        W1_size = input_size*hidden_size1
        W2_size = hidden_size1*hidden_size2

        self.W1 = np.array([[float(element)] for element in individual[0:W1_size]], dtype=np.single)
        self.W2 = np.array([[float(element)] for element in individual[W1_size:W1_size+W2_size]], dtype=np.single)
        self.W3 = np.array([[float(element)] for element in individual[W1_size+W2_size:]], dtype=np.single)

        self.W1 = self.W1.reshape([hidden_size1, input_size])
        self.W2 = self.W2.reshape([hidden_size2, hidden_size1])
        self.W3 = self.W3.reshape([output_size, hidden_size2])

        # Normalize
        # W1 = (W1 - W1.mean()) / W1.std()
        # W2 = (W2 - W2.mean()) / W2.std()
        # W3 = (W3 - W3.mean()) / W3.std()

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

        return input_size * hidden_size1 + hidden_size1 * hidden_size2 + hidden_size2 * output_size
