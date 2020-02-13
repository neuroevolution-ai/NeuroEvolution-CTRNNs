import gym
import pickle
import torch
import torch.nn as nn
import numpy as np
import time

# Fully connected neural network with one hidden layer
class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size1, hidden_size2, output_size, W1, W2, W3):
        super(NeuralNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size1)
        self.tanh1 = nn.Tanh()
        self.fc2 = nn.Linear(hidden_size1, hidden_size2)
        self.tanh2 = nn.Tanh()
        self.fc3 = nn.Linear(hidden_size2, output_size)
        self.tanh3 = nn.Tanh()

        self.W1, self.W2, self.W3 = W1, W2, W3
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

    def get_weight_matrizes(self):
        return self.W1, self.W2, self.W3


# Load halloffame candidate
with open("Weights_hof.pickle", "rb") as fp:
    W1, W2, W3 = pickle.load(fp)


# env = gym.make("InvertedPendulumSwingupBulletEnv-v0")
# env = gym.make("HalfCheetahBulletEnv-v0")
# env = gym.make("CartPoleBulletEnv-v1")
# env = gym.make("Walker2DBulletEnv-v0")
env = gym.make('Walker2d-v2')
env.render()
env.reset()

# test = gym.envs.registry.all()
fitness_current = 0
done = False


# Hyper-parameters
input_size = env.observation_space.shape[0]
hidden_size1 = 32
hidden_size2 = 16
output_size = env.action_space.shape[0]

print(input_size)
print(output_size)

model = NeuralNet(input_size, hidden_size1, hidden_size2, output_size, W1, W2, W3)

number_steps = 0

for i in range(10):

    fitness_current = 0
    ob = env.reset()
    done = False

    # Test fitness through simulation
    while not done:
        action = model.get_action(ob)
        # action = env.action_space.sample()
        ob, rew, done, info = env.step(action)
        number_steps = number_steps + 1
        fitness_current += rew

        env.render()
        time.sleep(0.0100)

    print(fitness_current)