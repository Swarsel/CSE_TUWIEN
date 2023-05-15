import numpy as np
from scipy.integrate import quad
import matplotlib.pyplot as plt


def fem_1d(N):
    # equidistant grid points
    h = 1 / N
    x_vals = np.arange(0, 1 + h, h)
    # vector of elements dependant on nodes
    dx = x_vals[1:] - x_vals[:-1]

    # stiffness matrix
    A = np.diag(1 / dx[:-1] + 1 / dx[1:])
    A = A - np.diag(1 / dx[1:-1], 1)
    A = A - np.diag(1 / dx[1:-1], -1)

    # load vector
    def left(x, a, b):
        return (2 / 9 * (x ** (-4 / 3) + 5 * x ** (-1 / 3))) * ((x - a) / (b - a))

    def right(x, a, b):
        return (2 / 9 * (x ** (-4 / 3) + 5 * x ** (-1 / 3))) * ((b - x) / (b - a))

    f1 = np.zeros(N - 1)
    f2 = np.zeros(N - 1)
    for i in range(1, N):
        f1[i - 1], _ = quad(lambda x: left(x, x_vals[i - 1], x_vals[i]), x_vals[i - 1], x_vals[i])
        f2[i - 1], _ = quad(lambda x: right(x, x_vals[i], x_vals[i + 1]), x_vals[i], x_vals[i + 1])
    f = (f1 + f2).T

    u = np.linalg.solve(A, f)
    u = np.concatenate(([0], u.flatten(), [0]))  # add boundaries

    x_exact = np.linspace(0, 1, 100)
    plt.title(f"FEM vs exact solution, N={N}")
    plt.plot(x_exact, x_exact ** (2 / 3) * (1 - x_exact), label="exact solution")
    plt.plot(x_vals, u, '-o', label="FEM")
    plt.xlabel('x')
    plt.ylabel('u(x)')
    plt.grid(True)
    plt.legend()
    plt.savefig(f"N={N}.png")
    # plt.show()
    plt.close()

    def du_l2(x):
        return (2 / 3 * x ** (-1 / 3) - 5 / 3 * x ** (2 / 3)) ** 2

    norm_u, _ = quad(lambda x: du_l2(x), 0, 1)
    norm_uN = u[1:-1].T @ f
    error = norm_u - norm_uN

    return error


hs = []
errors = []
for N in [5, 10, 15, 20, 25]:
    hs.append(1 / N)
    errors.append(fem_1d(N))

plt.semilogy(hs, errors, "-o", label="error")
plt.title("Errors in energy norm for FEM solution")
plt.xlabel('h')
plt.ylabel('error')
plt.grid(True)
plt.legend()
plt.savefig(f"energynorm.png")
plt.show()
