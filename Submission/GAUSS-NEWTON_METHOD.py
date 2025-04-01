import numpy as np
import matplotlib.pyplot as plt
from numpy.ma.core import zeros

# Load data from csv file
data = np.loadtxt('Optimization\i_t_i_Y_i.csv', delimiter=',',
                  skiprows=1)

# Split the cols into i, number of couples in the first gen, number of adults in the 2nd gen
i = data[:, 0]
t = data[:, 1]
Y = data[:, 2]

x1 = [200, 0.3, 0.5]
x2 = [120, 0.25, 0.5]
x3 = [180, 0.2, 0.7]


def f(t_i, x):
    x_1, x_2, x_3 = x[0], x[1], x[2]
    return x_1 * t_i * np.exp(-x_2 * t ** x_3)


f1 = f(t, x1)
f2 = f(t, x2)
f3 = f(t, x3)


# Defining the error
def r(x):
    return f(t, x) - Y


# Building the Jacobian
def Jacobian(x):
    # Grad of r
    r_x1 = t * np.exp(-x[1] * t ** x[2])  # w.r.t x1
    r_x2 = -(t ** x[2]) * f(t, x)  # w.r.t x2
    r_x3 = -x[1] * (t ** x[2]) * np.log(t) * f(t, x)  # w.r.t x3
    return np.array([r_x1, r_x2, r_x3]).transpose()


def grad_g(x):
    return 2 * np.matmul(Jacobian(x).transpose(),  r(x))

def GaussNewton(x):
    xk1 = x
    iter = 0
    while np.linalg.norm(grad_g(xk1)) > 1e-6:
        J = Jacobian(xk1)
        xk1 = xk1 - 0.5 * np.matmul(np.linalg.inv(np.matmul(J.T, J)), grad_g(xk1))
        iter += 1
    return xk1, iter


def Damped_GaussNewton(x, t):
    xk1 = x
    iter = 0
    while np.linalg.norm(grad_g(xk1)) > 1e-6:
        J = Jacobian(xk1)
        a = - np.linalg.inv(np.matmul(J.T, J))
        d = np.matmul(a, np.matmul(J.T, r(xk1)))
        xk1 = xk1 + t * d
        iter += 1
    return xk1, iter

print(Damped_GaussNewton(x2, .8))
print(Damped_GaussNewton(x2, .6))
print(Damped_GaussNewton(x2, .4))

# Creating the figure
fig, axes = plt.subplots(1, 2, figsize=(12, 6))
ax = axes[0]  # First axes
ax1 = axes[1]  # Second axes

# Scatter Plot
ax.scatter(t, Y)
ax.set_xlabel('Number of adults in the 2nd generation')
ax.set_ylabel('Number of couples in the 1st generation')
ax.set_title('Experimental Data')

# Plot of the model
ax1.plot(t, f1, label='$f(t_i;\\overrightarrow{x}_1)$')  # \\LaTeX is awesome, soo prettyy
ax1.plot(t, f2, label='$f(t_i;\\overrightarrow{x}_2)$')
ax1.plot(t, f3, label='$f(t_i;\\overrightarrow{x}_3)$')
ax1.set_title('Model of the Relationship')
ax1.set_xlabel('Number of couples in the 1st generation')
ax1.set_ylabel('Model for num. of adults in the 2nd generation')

# Making the plot prettier
plt.legend()
plt.tight_layout()
plt.show()
