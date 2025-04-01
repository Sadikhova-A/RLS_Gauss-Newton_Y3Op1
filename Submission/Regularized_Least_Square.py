import numpy as np
import matplotlib.pyplot as plt
from jedi.inference.imports import import_module_by_names
from scipy.interpolate import UnivariateSpline


# Load the data from the CSV file
data = np.loadtxt('Optimization/t_data_x_noisy.csv', delimiter=',', skiprows=1)

# Split the columns into time and noisy data
t_data = data[:, 0]
x_noisy = data[:, 1]

# Fit the noisy data using a spline to approximate x'(t)
spline_fit = UnivariateSpline(t_data, x_noisy, s=0.5)  # Smoothing spline
spline_derivative = spline_fit.derivative()  # Derivative of the spline

# Get the derivative values from the spline approximation
x_prime_approx = spline_derivative(t_data) # This is the approximation of x'(t)

x_true = np.exp(-t_data)


# Polynomial fitting
def poly_fitting(n, l):
    # A matrix whose rows are u_i = [1, u_i, u_i^2, ..., u_i^n]
    U = np.vander(t_data, n+1, increasing=True)  # Polynomial Fitting Matrix

    # Finding the Regularization Matrix
    D = np.zeros((n-1, n+1))
    for i in range(2, n+1):
        D[i-2][i] = i * (i-1)

    x_0 = np.matmul(U.transpose(), U) + l * np.matmul(D.transpose(), D)
    x_1 = np.linalg.inv(x_0)
    x_RLS = np.matmul(np.matmul(x_1, U.transpose()), x_prime_approx)
    return U, x_RLS


def f(n, l):
    U, x_RLS = poly_fitting(n, l)
    return np.matmul(U, x_RLS)


# Plotting
fig = plt.figure(figsize=(10, 8))
# Grid of figures
ax1 = fig.add_subplot(221)
ax2 = fig.add_subplot(222)
ax3 = fig.add_subplot(223)
ax4 = fig.add_subplot(224)
# Adding plots
ax1.plot(t_data, x_prime_approx, ls='--', c='gray', label="$x'(t)$ Approx")
ax2.plot(t_data, x_prime_approx, ls='--', c='gray', label="$x'(t)$ Approx")
ax3.plot(t_data, x_prime_approx, ls='--', c='gray', label="$x'(t)$ Approx")
ax4.plot(t_data, x_prime_approx, ls='--', c='gray', label="$x'(t)$ Approx")

# Added the noisy and derivative of true data to each axes for comparison
ax1.plot(t_data, x_noisy, c='cyan', label='$x(t)$ Noisy')
ax1.plot(t_data, -x_true, c='blue', label='$x(t) = -e^{-t}$')

ax2.plot(t_data,x_noisy, c='cyan', label='$x(t)$ Noisy')
ax2.plot(t_data, -x_true, c='blue', label='$x(t) = -e^{-t}$')

ax3.plot(t_data,x_noisy, c='cyan', label='$x(t)$ Noisy')
ax3.plot(t_data, -x_true, c='blue', label='$x(t) = -e^{-t}$')

ax4.plot(t_data,x_noisy, c='cyan', label='$x(t)$ Noisy')
ax4.plot(t_data, -x_true, c='blue', label='$x(t) = -e^{-t}$')

# Plotting the regularised square approximations for different lambda
ax1.plot(t_data, f(5, 0), c='purple', label='Polynomial Approx, $\\lambda = 0$')
ax2.plot(t_data, f(5, 1), c='green', label='Polynomial Approx, $\\lambda = 1$')
ax3.plot(t_data, f(5, 100), c='red', label='Polynomial Approx, $\\lambda = 100$')
ax4.plot(t_data, f(5, 10000), c='violet', label='Polynomial Approx, $\\lambda = 10000$')

# Making the axes prettier
ax1.set_title("$\\lambda = 0$")
ax2.set_title("$\\lambda = 1$")
ax3.set_title("$\\lambda = 100$")
ax4.set_title("$\\lambda = 10000$")

ax1.legend(fontsize="7", loc ="lower right")
ax2.legend(fontsize="7", loc ="lower right")
ax3.legend(fontsize="7", loc ="lower right")
ax4.legend(fontsize="7", loc ="lower right")
fig.tight_layout()
plt.show()