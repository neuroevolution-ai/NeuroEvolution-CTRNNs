import gym
import time

env = gym.make('MountainCarContinuous-v0')
env.render()
env.reset()

print(env.observation_space)
print(env.action_space)
temp1 = env.observation_space
temp2 = env.action_space

done = False
fitness = 0
while not done:
    ob, rew, done, info = env.step(env.action_space.sample()) # take a random action
    env.render()
    fitness += rew
    print(rew)
    #print(done)
    time.sleep(0.100)

print(fitness)
env.close()

