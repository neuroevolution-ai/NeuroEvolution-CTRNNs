
import numpy as np


class EpisodeRunner(object):

    def __init__(self, conf, discrete_actions, brain_class, input_size, output_size, env):
        self.conf = conf
        self.discrete_actions = discrete_actions
        self.brain_class = brain_class
        self.input_size = input_size
        self.output_size = output_size
        self.env = env

    def evalFitness(self, individual):
        brain = self.brain_class(self.input_size, self.output_size, individual, self.conf)
        fitness_current = 0
        number_fitness_runs = self.conf["number_fitness_runs"]

        for i in range(number_fitness_runs):

            ob = self.env.reset()
            done = False
            consecutive_non_movement = 0
            while not done:

                # Perform step of the brain simulation
                action = brain.step(ob)
                if self.discrete_actions:
                    action = np.argmax(action)
                # Perform step of the environment simulation
                ob, rew, done, info = self.env.step(action)

                if self.conf["environment"] == "BipedalWalker-v3":
                    if ob[2] < 0.0001:
                        consecutive_non_movement = consecutive_non_movement + 1
                        if consecutive_non_movement > 50:
                            done = True
                            rew = rew - 300
                    else:
                        consecutive_non_movement = 0

                fitness_current += rew

        return fitness_current / number_fitness_runs,
