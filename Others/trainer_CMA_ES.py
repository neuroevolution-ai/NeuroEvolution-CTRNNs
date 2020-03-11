from deap import base
from deap import creator
from deap import tools
from deap import cma
from deap import algorithms


class TrainerCmaEs(object):
    def __init__(self, evalFitness, individual_size, conf, map_func=map, hof=tools.HallOfFame(5),
                 trainer_parameters=None):
        self.toolbox = toolbox = base.Toolbox()
        self.conf = conf
        self.hof = hof
        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
        creator.create("Individual", list, typecode='b', fitness=creator.FitnessMax)
        toolbox.register("map", map_func)
        toolbox.register("evaluate", evalFitness)
        strategy = cma.Strategy(centroid=[0.0] * individual_size, sigma=conf["sigma"],
                                lambda_=conf["population_size"])

        toolbox.register("generate", strategy.generate, creator.Individual)
        toolbox.register("update", strategy.update)

    def train(self, stats, number_generations):
        return algorithms.eaGenerateUpdate(self.toolbox, ngen=number_generations,
                                           stats=stats, halloffame=self.hof)
