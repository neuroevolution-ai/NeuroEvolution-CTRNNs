from deap import base
from deap import creator
from deap import tools
from deap import cma
from deap import algorithms
import numpy as np


def sel_elitist_tournament(individuals, mu, k_elitist, k_tournament, tournsize):
    return tools.selBest(individuals, k_elitist) + tools.selTournament(individuals, k_tournament, tournsize=tournsize)


class TrainerMuPlusLambda(object):
    def __init__(self, evalFitness, individual_size, conf, map_func=map, hof=tools.HallOfFame(5)):
        self.toolbox = toolbox = base.Toolbox()
        self.conf = configuration_data = conf
        self.hof = hof

        toolbox.register("map", map_func)
        toolbox.register("evaluate", evalFitness)

        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
        creator.create("Individual", list, typecode='b', fitness=creator.FitnessMax)
        toolbox.register("indices", np.random.uniform, -20, 20, individual_size)

        toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.indices)

        toolbox.register("population", tools.initRepeat, list, toolbox.individual)
        toolbox.register("mate", tools.cxOnePoint)
        toolbox.register("mutate", tools.mutGaussian, mu=0.0, sigma=5, indpb=0.2)

        toolbox.register("select",
                         sel_elitist_tournament,
                         k_elitist=int(0.1 * configuration_data["population_size"]),
                         k_tournament=configuration_data["population_size"] - int(
                             0.1 * configuration_data["population_size"]),
                         tournsize=2)

    def train(self, stats):
        pop = self.toolbox.population(n=int(self.conf["population_size"] * 0.4))
        return algorithms.eaMuPlusLambda(pop,
                                         toolbox=self.toolbox,
                                         ngen=self.conf["number_generations"],
                                         stats=stats,
                                         mu=int(self.conf["population_size"] * .5),
                                         lambda_=int(self.conf["population_size"] * 1.),
                                         cxpb=0.2, mutpb=0.8,
                                         halloffame=self.hof
                                         )
