import numpy as np


class ContinuousTimeRNN:

    def __init__(self, name):
        self.name = name

    def continuous_time_rnn(self, y, alpha, V, W, u):
        u2 = u[:, np.newaxis]

        # Differential equation
        dydt = np.dot(W, np.tanh(y)) + np.dot(V, u2)

        return dydt