import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt


def model1_np(y,t, alpha, V, W):

    u=1

    # Differential equation
    # dydt = -alpha*y + np.dot(W, np.tanh(y + np.dot(V,u)))
    # dydt = np.dot(W, np.tanh(y + np.dot(V,u)))
    dydt = -alpha*y + np.tanh( np.dot(W,y) + np.dot(V,u) )
    # dydt = np.dot(W, np.tanh(y)) + np.dot(V, u)

    return dydt


# function that returns dy/dt
def model1(y_list,t, alpha, V, W):

    # Convert list to 2D numpy array
    y = np.array([[element] for element in y_list])

    dydt = model1_np(y,t, alpha, V, W)

    return [element[0] for element in dydt]

def model2(y,t):

    u=1
    dy1dt = y[1]
    dy2dt = t*y[1] + np.sin(y[0]) + u

    return [dy1dt, dy2dt]


# Number of neurons
N = 5

# Alpha
alpha = 0.01

# Time vector
t = np.linspace(0,1,100)

# Solve ODE for target function
#y_target_function = odeint(model2, [0,0], t)
#y1_target_function = y_target_function[:,0]
y1_target_function = np.sin(2*t)

# Convert 1D array to a column vector.
y1_target_function = y1_target_function[:, np.newaxis]

k = 0

difference_min = np.inf

min_value = -1.0
max_value = 1.0

# Initialize weight matrizes
V = np.random.uniform(min_value, max_value, (N, 1))
W = np.random.uniform(min_value, max_value, (N, N))
T = np.random.uniform(min_value, max_value, (N, 1))

# Solve ODE for neural network
while True:

    # Weight matrizes
    V_new = np.array(V, copy=True)
    W_new = np.array(W, copy=True)
    T_new = np.array(T, copy=True)

    # Randomly update any index
    m = np.random.randint(N+N+N*N)
    if m < N:
        V_new[np.random.randint(V_new.shape[0]), np.random.randint(V_new.shape[1])] = np.random.uniform(min_value, max_value)
    elif m < 2*N:
        T_new[np.random.randint(T_new.shape[0]), np.random.randint(T_new.shape[1])] = np.random.uniform(min_value, max_value)
    else:
        W_new[np.random.randint(W_new.shape[0]), np.random.randint(W_new.shape[1])] = np.random.uniform(min_value, max_value)

    # Solve ODE
    y_neural_network = odeint(model1, [0]*N, t, args=(alpha, V_new, W_new))

    # Calculate outputs o
    o = np.dot(y_neural_network, T_new)

    difference = np.sum(np.abs(o-y1_target_function))
    print(difference)

    k=k+1

    if difference < difference_min:
        V = V_new
        W = W_new
        T = T_new
        difference_min = difference
        print(difference)

    if difference < 0.3:
        break

print(V)
print(W)
print(T)
print(k)
print(difference)

# Anfangswerte
y = np.zeros(N)
y = y[:, np.newaxis]

o_discrete = []
# Calculate discrete Time steps
delta_t = t[1] - t[0]
for t_i in t:
    dy = model1_np(y, t_i, alpha, V, W)

    y = y + delta_t*dy

    o_i = np.dot(y.T, T)

    o_discrete.append(o_i[0][0])

# Plot results
plt.plot(t,o,'b--')
plt.plot(t,o_discrete,'g--')
plt.plot(t,y1_target_function,'r-')
plt.xlabel('time')
plt.legend(['Continous-Time RNN', 'Discrete-Time RNN', 'Target Function'])
plt.grid()
plt.show()

print("finished")
