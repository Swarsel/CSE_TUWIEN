import numpy as np
import scipy as sp
from scipy.integrate import quad
import matplotlib.pyplot as plt


def fem_1d(nodes):
    N = len(nodes) - 1
    # vector of elements dependant on nodes
    dx = nodes[1:] - nodes[:-1]

    # stiffness matrix - this sadly quickly starts to eat a lot of memory
    # i also tried to set it manually elementwise - but it is a lot faster and we also quickly reach
    # max array size
    A = np.diag(1 / dx[:-1] + 1 / dx[1:])
    A = A - np.diag(1 / dx[1:-1], 1)
    A = A - np.diag(1 / dx[1:-1], -1)

    A = sp.sparse.csc_matrix(A) # sparsify to save a bit of time

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

    u = sp.sparse.linalg.spsolve(A, f)  # switched to scipy (seems more efficient)
    u = np.concatenate(([0], u.flatten(), [0]))  # add boundaries

    return A, f, u


def energy_norm(u, f):
    # integral gives 6/7, so skip expensive calculation
    # def du_l2(x):
    #    return (2 / 3 * x ** (-1 / 3) - 5 / 3 * x ** (2 / 3)) ** 2

    # norm_u, _ = quad(lambda x: du_l2(x), 0, 1)
    norm_uN = u[1:-1].T @ f
    # error = norm_u - norm_uN
    error = 6 / 7 - norm_uN
    return error


def estimate(u, nodes):
    def f_squared(x):
        # return (2 / 9 * (x ** (-4 / 3) + 5 * x ** (-1 / 3))) ** 2
        return (4 * (5 * x + 1) ** 2) / (
                81 * x ** (8 / 3))  # simplified expression from WolframAlpha to maybe save time

    dx = nodes[1:] - nodes[:-1]
    midpoints = 0.5 * (nodes[1:] + nodes[:-1])

    f_mid = f_squared(midpoints)
    f_norm = dx * f_mid

    u_diff = (u[1:] - u[:-1]) / dx

    # p.54
    normal_jump = (u_diff[1:] - u_diff[:-1]) ** 2  # differences of u in nodal values
    normal_jump = np.concatenate(([0], normal_jump.flatten(), [0]))  # add boundaries
    normal_jump = normal_jump[1:] + normal_jump[:-1]  # sum of adjacent derivatives
    res_error = dx ** 2 * f_norm + dx * normal_jump
    return res_error


# marking parameter and tolerance for each - i run out for memory if i dont adjust it
# also added are the errors in the energy norm that i received for the same size problem in 8.1
for theta, tol, ex1 in zip([0.25, 1], [0.005, 0.05], [0.0577316664423978, 0.019631474377605374]):
    h = 0.2
    x_vals = np.arange(0, 1 + h, h)

    no_elements = []
    errors = []
    etas = []

    while True:
        # SOLVE
        A, f, u = fem_1d(x_vals)

        # ESTIMATE
        eta_T = estimate(u, x_vals)  # get local error contributions
        idx = np.argsort(eta_T)[::-1]  # descending list of sort indices
        eta = np.cumsum(eta_T[idx])  # list of cumulative sums of local error estimators for dörfler marking
        sum_eta = eta[-1]

        if sum_eta <= tol:
            break

        # MARK
        marked = idx[:np.argmax(eta >= theta * sum_eta) + 1]  # dörfler marking, get indices of selected elements
        marked_sorted = np.sort(marked)

        # REFINE
        for i in range(len(marked_sorted)):
            # we add +i to each index because we are adding an element in each iteration
            x_vals = np.concatenate((x_vals[:marked_sorted[i] + i + 1],
                                     0.5 * (x_vals[marked_sorted[i] + i] + x_vals[marked_sorted[i] + i + 1]),
                                     x_vals[marked_sorted[i] + i + 1:]), axis=None)

        no_elements.append(x_vals.size)
        errors.append(energy_norm(u, f))
        etas.append(sum_eta)

    plt.loglog(no_elements, errors, label="error")
    plt.loglog(no_elements, etas, label="$\\eta$")
    plt.title(f"error and indicator over $N$, $\\theta = {theta}$")
    plt.xlabel("N")
    plt.legend()
    plt.grid()
    plt.savefig(f"error_indicator_theta={theta}.png")
    plt.show()
    plt.close()

    x_exact = np.linspace(0, 1, 100)
    plt.title(f"FEM vs exact solution, N={no_elements[-1]}, theta={theta}")
    plt.plot(x_exact, x_exact ** (2 / 3) * (1 - x_exact), label="exact solution")
    plt.plot(x_vals, u, '-o', label="FEM")
    plt.xlabel('x')
    plt.ylabel('u(x)')
    plt.grid(True)
    plt.legend()
    plt.savefig(f"fem_vs_exact_theta={theta}.png")
    plt.show()
    plt.close()

    print(f"N={no_elements[-1]}, error={errors[-1]}, error of equivalent size FEM of ex1 was {ex1}")
