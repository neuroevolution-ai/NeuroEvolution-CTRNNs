import random
import time
import numpy as np

from deap import algorithms
from deap import base
from deap import creator
from deap import tools
from scipy.integrate import odeint
import matplotlib.pyplot as plt

from scoop import futures


creator.create("FitnessMax", base.Fitness, weights=(-1.0,))
creator.create("Individual", list, typecode='b', fitness=creator.FitnessMax)

# Number of neurons
N = 5

# Alpha
alpha = 0.01


# Size of Individual
IND_SIZE = N+N+N*N

toolbox = base.Toolbox()
toolbox.register("attr_float", random.uniform, -1, 1)
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_float, n=IND_SIZE)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

# Multiprocessing
toolbox.register("map", futures.map)


def model1_np(y,t, alpha, V, W):

    u=1

    # Differential equation
    # dydt = -alpha*y + np.dot(W, np.tanh(y + np.dot(V,u)))
    # dydt = np.dot(W, np.tanh(y + np.dot(V,u)))
    dydt = -alpha*y + np.tanh(np.dot(W,y) + np.dot(V,u))
    # dydt = np.tanh(np.dot(W, y) + np.dot(V, u))
    # dydt = np.dot(W, np.tanh(y)) + np.dot(V, u)

    return dydt


# function that returns dy/dt
def model1(y_list,t, alpha, V, W):

    # Convert list to 2D numpy array
    y = np.array([[element] for element in y_list])

    dydt = model1_np(y,t, alpha, V, W)

    return [element[0] for element in dydt]


def evalFitness(individual):

    # Get weight matrizes
    V = np.array([[element] for element in individual[0:N]])
    T = np.array([[element] for element in individual[N:2*N]])
    W = np.array([[element] for element in individual[2*N:]])
    W = W.reshape([N,N])

    # Time vector
    t = np.linspace(0, 1, 100)

    # Target function
    y1_target_function = np.sin(2 * t)

    # Convert 1D array to a column vector.
    y1_target_function = y1_target_function[:, np.newaxis]

    # Solve ODE
    y_neural_network = odeint(model1, [0]*N, t, args=(alpha, V, W))

    # Calculate outputs o
    o = np.dot(y_neural_network, T)

    difference = np.sum(np.abs(o-y1_target_function))

    return difference,


toolbox.register("mate", tools.cxUniform, indpb=0.05)
#toolbox.register("mate", tools.cxTwoPoint)
toolbox.register("mutate", tools.mutGaussian, mu=0.1, sigma=0.3, indpb=0.3)
toolbox.register("select", tools.selTournament, tournsize=10)
toolbox.register("evaluate", evalFitness)

# Time vector
t = np.linspace(0,1,100)

# Target function
y1_target_function = np.sin(2 * t)

# Convert 1D array to a column vector.
y1_target_function = y1_target_function[:, np.newaxis]

if __name__ == "__main__":

    startTime = time.time()

    pop = toolbox.population(n=1500)
    hof = tools.HallOfFame(5)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("std", np.std)
    stats.register("min", np.min)
    stats.register("max", np.max)

    pop, log = algorithms.eaSimple(pop, toolbox, cxpb=0.2, mutpb=0.2, ngen=300, stats=stats, halloffame=hof, verbose=True)

    print("Time elapsed: %s" % (time.time() - startTime))

    # Get best individual
    best_individual = hof[0]

    # Get weight matrizes of best individual
    V_best = np.array([[element] for element in best_individual[0:N]])
    T_best = np.array([[element] for element in best_individual[N:2*N]])
    W_best = np.array([[element] for element in best_individual[2*N:]])
    W_best = W_best.reshape([N,N])

    # Solve ODE
    y_neural_network = odeint(model1, [0]*N, t, args=(alpha, V_best, W_best))

    # Calculate outputs o
    o = np.dot(y_neural_network, T_best)

    # Anfangswerte
    y = np.zeros(N)
    y = y[:, np.newaxis]

    o_discrete = []
    # Calculate discrete Time steps
    delta_t = t[1] - t[0]
    for t_i in t:
        dy = model1_np(y, t_i, alpha, V_best, W_best)

        y = y + delta_t * dy

        o_i = np.dot(y.T, T_best)

        o_discrete.append(o_i[0][0])

    # Print difference
    print("Difference: %s" % np.sum(np.abs(o - y1_target_function)))

    # Plot results
    plt.plot(t, o, 'b--')
    plt.plot(t, o_discrete, 'g--')
    plt.plot(t, y1_target_function, 'r-')
    plt.xlabel('time')
    plt.legend(['Continous-Time RNN', 'Discrete-Time RNN', 'Target Function'])
    plt.grid()
    plt.show()