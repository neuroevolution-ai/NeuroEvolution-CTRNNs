import mujoco_py
import gym
import time
import numpy as np
import json

env = gym.make('InvertedPendulum-v2')
# env = gym.make('BipedalWalker-v3')
#env = gym.make('MountainCarContinuous-v0')
env.render()
env.reset()

temp1 = env.observation_space.shape[0]
temp2 = env.action_space.shape[0]
print(temp1)
print(temp2)

with open("Configuration_Optimization.json", "r") as read_file:
    data = json.load(read_file)

print(data["environment"])
print(data["number_neurons"])
print(data["delta_t"])
print(data["population_size"])
print(data["number_generations"])
print(data["number_fitness_runs"])

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

