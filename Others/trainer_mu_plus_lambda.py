from deap import base
from deap import creator
from deap import tools
from deap import cma
from Others import algorithms
import numpy as np


def sel_elitist_tournament(individuals, mu, k_elitist, k_tournament, tournsize):
    return tools.selBest(individuals, int(k_elitist*mu)) + \
           tools.selTournament(individuals, int(k_tournament * mu), tournsize=tournsize)


class TrainerMuPlusLambda(object):
    def __init__(self, evalFitness, individual_size, trainer_parameters, population_size, map_func=map,
                 hof=tools.HallOfFame(5), checkpoint=None):
        self.toolbox = toolbox = base.Toolbox()
        self.conf = trainer_parameters
        self.hof = hof
        self.population_size = population_size

        toolbox.register("map", map_func)
        toolbox.register("evaluate", evalFitness)

        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
        creator.create("Individual", list, typecode='b', fitness=creator.FitnessMax)
        toolbox.register("indices", np.random.uniform,
                         -self.conf["initial_gene_range"],
                         self.conf["initial_gene_range"],
                         individual_size)

        toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.indices)

        toolbox.register("population", tools.initRepeat, list, toolbox.individual)

        if self.conf["mate"] == "cxOnePoint":
            toolbox.register("mate", tools.cxOnePoint)
        elif self.conf["mate"] == "cxTwoPoint":
            toolbox.register("mate", tools.cxTwoPoint)
        elif self.conf["mate"] == "cxUniform":
            toolbox.register("mate", tools.cxUniform, indpb=self.conf["mate_indpb"])
        else:
            raise RuntimeError("unknown mate function")

        toolbox.register("mutate", tools.mutGaussian,
                         mu=0.0,
                         sigma=self.conf["mutation_Gaussian_sigma"],
                         indpb=self.conf["mutation_Gaussian_indpb"])

        toolbox.register("select",
                         sel_elitist_tournament,
                         k_elitist=int(self.conf["elitist_ratio"]),
                         k_tournament=1.0 - int(
                             self.conf["elitist_ratio"]),
                         tournsize=self.conf["tournsize"])

    def train(self, stats, number_generations, checkpoint=None):
        pop = self.toolbox.population(n=int(self.population_size))
        return algorithms.eaMuPlusLambda(pop,
                                         toolbox=self.toolbox,
                                         ngen=number_generations,
                                         stats=stats,
                                         mu=int(self.population_size * self.conf["mu"]),
                                         lambda_=int(self.population_size * self.conf["lambda"]),
                                         cxpb=1.0-self.conf["mutpb"], mutpb=self.conf["mutpb"],
                                         halloffame=self.hof,
                                         checkpoint=checkpoint
                                         )
