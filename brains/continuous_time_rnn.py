import numpy as np


class ContinuousTimeRNN:

    def __init__(self, individual, input_size, number_neurons, output_size, optimize_y0, delta_t, clipping_range_min,
                 clipping_range_max, set_principle_diagonal_elements_of_W_negative):

        V_size = input_size * number_neurons
        W_size = number_neurons * number_neurons
        T_size = number_neurons * output_size
        M_size = V_size + W_size + T_size

        # Anfangswerte für y
        if optimize_y0:
            y0 = np.array([element for element in individual[M_size:M_size+number_neurons]])
        else:
            y0 = np.zeros(number_neurons)

        self.y = y0[:, np.newaxis]
        self.clipping_range_min = clipping_range_min
        self.clipping_range_max = clipping_range_max
        self.delta_t = delta_t
        self.set_principle_diagonal_elements_of_W_negative = set_principle_diagonal_elements_of_W_negative

        # Get weight matrizes of current individual
        self.V = np.array([[element] for element in individual[0:V_size]])
        self.W = np.array([[element] for element in individual[V_size:V_size + W_size]])
        self.T = np.array([[element] for element in individual[V_size + W_size:V_size + W_size + T_size]])

        self.V = self.V.reshape([number_neurons, input_size])
        self.W = self.W.reshape([number_neurons, number_neurons])
        self.T = self.T.reshape([number_neurons, output_size])

        self.clipping_range_min = [-abs(element) for element in individual[M_size+number_neurons:M_size+2*number_neurons]]
        self.clipping_range_max = [abs(element) for element in individual[M_size+2*number_neurons:]]

        # Set elements of main diagonal to less than 0
        if set_principle_diagonal_elements_of_W_negative:
            for j in range(number_neurons):
                self.W[j][j] = -abs(self.W[j][j])

    def step(self, ob):

        u = ob[:, np.newaxis]

        # Differential equation
        dydt = np.dot(self.W, np.tanh(self.y)) + np.dot(self.V, u)

        # Euler forward discretization
        self.y = self.y + self.delta_t * dydt

        # Clip y to state boundaries
        for y_min, y_max in zip(self.clipping_range_min, self.clipping_range_max):
            self.y = np.clip(self.y, y_min, y_max)

        self.y = np.clip(self.y, self.clipping_range_min, self.clipping_range_max)

        # Calculate outputs
        o = np.tanh(np.dot(self.y.T, self.T))

        return o[0]

    @staticmethod
    def get_individual_size(input_size, number_neurons, output_size, optimize_y0):

        individual_size = input_size * number_neurons + number_neurons * number_neurons + number_neurons * output_size

        if optimize_y0:
            individual_size += number_neurons

        # For optimizing clipping ranges
        individual_size += 2* number_neurons

        return individual_size
