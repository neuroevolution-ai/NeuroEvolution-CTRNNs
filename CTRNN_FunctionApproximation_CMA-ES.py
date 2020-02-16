import random
import time
import numpy as np

from deap import algorithms
from deap import base
from deap import creator
from deap import tools
from deap import cma
from scoop import futures

from scipy.integrate import odeint
import matplotlib.pyplot as plt


def model1_np(y,t, alpha, V, W):

    u=1

    # Differential equation
    # dydt = -alpha*y + np.dot(W, np.tanh(y + np.dot(V,u)))
    # dydt = np.dot(W, np.tanh(y + np.dot(V,u)))
    # dydt = -alpha*y + np.tanh(np.dot(W,y) + np.dot(V,u))
    # dydt = np.tanh(np.dot(W, y) + np.dot(V, u))
    dydt = np.dot(W, np.tanh(y)) + np.dot(V, u)

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


# Number of neurons
N = 30

# Alpha
alpha = 0.01

# Size of Individual
IND_SIZE = N+N+N*N

print(IND_SIZE)

creator.create("FitnessMax", base.Fitness, weights=(-1.0,))
creator.create("Individual", list, typecode='b', fitness=creator.FitnessMax)

toolbox = base.Toolbox()

# Multiprocessing
toolbox.register("map", futures.map)

toolbox.register("evaluate", evalFitness)
strategy = cma.Strategy(centroid=[0.0] * IND_SIZE, sigma=0.1, lambda_=500)
# strategy = cma.Strategy(centroid=[0.0] * IND_SIZE, sigma=5.0)
toolbox.register("generate", strategy.generate, creator.Individual)
toolbox.register("update", strategy.update)
 
# Time vector
t = np.linspace(0,1,100)

# Target function
y1_target_function = np.sin(2 * t)

# Convert 1D array to a column vector.
y1_target_function = y1_target_function[:, np.newaxis]

if __name__ == "__main__":

    startTime = time.time()

    hof = tools.HallOfFame(5)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("std", np.std)
    stats.register("min", np.min)
    stats.register("max", np.max)

    pop, log = algorithms.eaGenerateUpdate(toolbox, ngen=1500, stats=stats, halloffame=hof)

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

    # Calculate continous outputs o
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