from deap import base
from deap import creator
from deap import tools
from deap import cma
from Others import algorithms
import numpy as np
import random
from functools import partial


def sel_elitist_tournament(individuals, mu, k_elitist, k_tournament, tournsize):
    return tools.selBest(individuals, int(k_elitist * mu)) + \
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

        mate_list = [
            tools.cxOnePoint,
            tools.cxTwoPoint,
            partial(tools.cxUniform, indpb=self.conf["mate_indpb_1"]),
            partial(tools.cxUniform, indpb=self.conf["mate_indpb_2"])
        ]

        if self.conf["mutation_Gaussian_dynamic_prob"]:
            mut_list = [partial(tools.mutGaussian,
                                mu=0.0,
                                sigma=self.conf["mutation_Gaussian_sigma_base"] ** (
                                            -random.random() * self.conf["mutation_Gaussian_sigma_factor"]),
                                indpb=self.conf["mutation_Gaussian_indpb_base"] ** (
                                        -random.random() * self.conf["mutation_Gaussian_indpb_factor"]))]
        else:
            mut_list = [
                partial(tools.mutGaussian,
                        mu=0.0,
                        sigma=self.conf["mutation_Gaussian_sigma_1"],
                        indpb=self.conf["mutation_Gaussian_indpb_1"]),
                partial(tools.mutGaussian,
                        mu=0.0,
                        sigma=self.conf["mutation_Gaussian_sigma_2"],
                        indpb=self.conf["mutation_Gaussian_indpb_2"]),
                partial(tools.mutGaussian,
                        mu=0.0,
                        sigma=self.conf["mutation_Gaussian_sigma_3"],
                        indpb=self.conf["mutation_Gaussian_indpb_3"])
            ]

        def mate(ind1, ind2):
            return random.choice(mate_list)(ind1, ind2)

        def mutate(ind1):
            return random.choice(mut_list)(ind1)

        toolbox.register("mate", mate)

        toolbox.register("mutate", mutate)

        toolbox.register("select",
                         sel_elitist_tournament,
                         k_elitist=int(self.conf["elitist_ratio"]),
                         k_tournament=1.0 - int(
                             self.conf["elitist_ratio"]),
                         tournsize=self.conf["tournsize"])

    def train(self, stats, number_generations, checkpoint=None, cb_before_each_generation=None):
        pop = self.toolbox.population(n=int(self.population_size))
        return algorithms.eaMuPlusLambda(pop,
                                         toolbox=self.toolbox,
                                         ngen=number_generations,
                                         stats=stats,
                                         mu=int(self.population_size * self.conf["mu"]),
                                         lambda_=int(self.population_size * self.conf["lambda"]),
                                         cxpb=1.0 - self.conf["mutpb"], mutpb=self.conf["mutpb"],
                                         halloffame=self.hof,
                                         checkpoint=checkpoint,
                                         cb_before_each_generation=cb_before_each_generation,
                                         include_parents_in_next_generation=self.conf[
                                             "include_parents_in_next_generation"]
                                         )
