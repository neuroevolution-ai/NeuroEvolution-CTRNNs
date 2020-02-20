import gym
import pickle
import torch
import torch.nn as nn
import numpy as np
import time

# Fully connected neural network with one hidden layer
class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size1, hidden_size2, output_size, individual, indirect_encoding=False):
        super(NeuralNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size1, bias=False)
        self.tanh1 = nn.Tanh()
        self.fc2 = nn.Linear(hidden_size1, hidden_size2, bias=False)
        self.tanh2 = nn.Tanh()
        self.fc3 = nn.Linear(hidden_size2, output_size, bias=False)
        self.tanh3 = nn.Tanh()

        self.input_size = input_size
        self.hidden_size1 = hidden_size1
        self.hidden_size2 = hidden_size2
        self.output_size = output_size

        self._set_weight_matrizes(individual, indirect_encoding)

        self.fc1.weight.data = torch.from_numpy(self.W1)
        self.fc2.weight.data = torch.from_numpy(self.W2)
        self.fc3.weight.data = torch.from_numpy(self.W3)

    def forward(self, x):
        out = self.fc1(x)
        out = self.tanh1(out)
        out = self.fc2(out)
        out = self.tanh2(out)
        out = self.fc3(out)
        out = self.tanh3(out)
        return out

    def get_action(self, ob):
        ob2 = ob[np.newaxis, :]
        ob2 = ob2.astype(np.single)
        output = self(torch.from_numpy(ob2))
        output_np = output.detach().numpy()[0, :]

        return output_np

    def _set_weight_matrizes(self, individual, indirect_encoding):

        W1_size = self.input_size*self.hidden_size1
        W2_size = self.hidden_size1*self.hidden_size2
        W3_size = self.hidden_size2*self.output_size

        # Indirect encoding
        if indirect_encoding:
            pass

        # Direct encoding
        else:
            # Weight Matrizes
            self.W1 = np.array([[float(element)] for element in individual[0:W1_size]], dtype=np.single)
            self.W2 = np.array([[float(element)] for element in individual[W1_size:W1_size+W2_size]], dtype=np.single)
            self.W3 = np.array([[float(element)] for element in individual[W1_size+W2_size:W1_size+W2_size+W3_size]], dtype=np.single)

            # Bias Matrizes
            index_b = W1_size + W2_size + W3_size
            self.B1 = np.array([float(element) for element in individual[index_b:index_b+self.hidden_size1]], dtype=np.single)
            self.B2 = np.array([float(element) for element in individual[index_b+self.hidden_size1:index_b+self.hidden_size1+self.hidden_size2]], dtype=np.single)
            self.B3 = np.array([float(element) for element in individual[index_b+self.hidden_size1+self.hidden_size2:]], dtype=np.single)

            self.W1 = self.W1.reshape([self.hidden_size1, self.input_size])
            self.W2 = self.W2.reshape([self.hidden_size2, self.hidden_size1])
            self.W3 = self.W3.reshape([self.output_size, self.hidden_size2])

            # Normalize
            #self.W1 = (self.W1 - self.W1.mean()) / max(self.W1.std(), 0.1)
            #self.W2 = (self.W2 - self.W2.mean()) / max(self.W2.std(), 0.1)
            #self.W3 = (self.W3 - self.W3.mean()) / max(self.W3.std(), 0.1)
            #self.B1 = (self.B1 - self.B1.mean()) / max(self.B1.std(), 0.1)
            #self.B2 = (self.B2 - self.B2.mean()) / max(self.B2.std(), 0.1)
            #self.B3 = (self.B3 - self.B3.mean()) / max(self.B3.std(), 0.1)

    def get_weight_matrizes(self):
        return self.W1, self.W2, self.W3

# Load halloffame candidate
with open("Weights_hof.pickle", "rb") as fp:
    individual = pickle.load(fp)

env = gym.make('Acrobot-v1')
env.render()
env.reset()

# test = gym.envs.registry.all()
fitness_current = 0
done = False


# Hyper-parameters
input_size = env.observation_space.shape[0]
hidden_size1 = 8
hidden_size2 = 4
# output_size = env.action_space.shape[0]
output_size = 3

print(input_size)
print(output_size)

model = NeuralNet(input_size, hidden_size1, hidden_size2, output_size, individual, indirect_encoding=False)

number_steps = 0

for i in range(10):

    fitness_current = 0
    ob = env.reset()
    done = False

    # Test fitness through simulation
    while not done:
        action2 = model.get_action(ob)
        # action = env.action_space.sample()

        action = np.argmax(action2)

        ob, rew, done, info = env.step(action)
        number_steps = number_steps + 1
        fitness_current += rew

        env.render()
        time.sleep(0.100)

    print(fitness_current)