import mujoco_py
import gym
import time
import numpy as np
import json
from datetime import datetime
import os

env = gym.make('InvertedPendulum-v2')
env.render()
env.reset()

temp1 = env.observation_space.shape[0]
temp2 = env.action_space.shape[0]
print(temp1)
print(temp2)

with open("Configuration.json", "r") as read_file:
    data = json.load(read_file)

# Create new directory for simulation run
directory = os.path.join('Simulation-Results', datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
os.makedirs(directory)

# Save json
with open(os.path.join(directory, 'Configuration.json'), 'w') as outfile:
    json.dump(data, outfile)

done = False
fitness = 0

while not done:
    temp = env.action_space.sample()
    ob, rew, done, info = env.step(env.action_space.sample()) # take a random action
    env.render()
    fitness += rew
    # print(rew)
    #print(done)
    # time.sleep(0.100)

print(fitness)
env.close()

