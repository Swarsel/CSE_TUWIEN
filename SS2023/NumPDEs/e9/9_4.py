import numpy as np
import scipy as sp
from scipy.integrate import quad
import matplotlib.pyplot as plt


def stiffness_matrix(nodes):
    dx = nodes[1:] - nodes[:-1]

    A = np.diag(1 / dx[:-1] + 1 / dx[1:])
    A = A - np.diag(1 / dx[1:-1], 1)
    A = A - np.diag(1 / dx[1:-1], -1)

    A = sp.sparse.csc_matrix(A)
    return A


def mass_matrix(h):
    N = int(1 / h) + 1

    M = (2 * h / 3) * np.diag(np.ones(N - 2))
    M = M + (h / 6) * np.diag(np.ones(N - 3), 1)
    M = M + (h / 6) * np.diag(np.ones(N - 3), -1)
    M = sp.sparse.csc_matrix(M)
    return M


def load_vector(nodes, t):
    def left(x, a, b):
        return (np.exp(-t) * (x ** 2 - x + 2)) * ((x - a) / (b - a))

    def right(x, a, b):
        return (np.exp(-t) * (x ** 2 - x + 2)) * ((b - x) / (b - a))

    f1 = np.zeros(len(nodes) - 2)
    f2 = np.zeros(len(nodes) - 2)
    for i in range(1, len(nodes) - 1):
        f1[i - 1], _ = quad(lambda x: left(x, nodes[i - 1], nodes[i]), nodes[i - 1], nodes[i])
        f2[i - 1], _ = quad(lambda x: right(x, nodes[i], nodes[i + 1]), nodes[i], nodes[i + 1])
    f = (f1 + f2).T
    return f


def implicit_euler(nodes, ts, h, k):
    u_0 = u_exact(nodes, 0)
    A = stiffness_matrix(nodes)
    M = mass_matrix(h)
    u = np.zeros((len(nodes), len(ts)))
    u[:, 0] = u_0
    col = 0
    for t in ts[1:]:
        u[1:-1, col + 1] = sp.sparse.linalg.spsolve(M + k * A, M * u[1:-1, col] + k * load_vector(nodes, t))
        col += 1
    return u


def u_exact(x, t):
    return np.exp(-t) * x * (1 - x)


# differing c
h = 0.1
for c in [0.5, 1, 1.1]:
    k = c * h
    ts = np.arange(0, 1 + k, k)
    nodes = np.arange(0, 1 + h, h)

    u = implicit_euler(nodes, ts, h, k)
    xs = np.linspace(0, 1, 100)
    plt.title(f"exact vs implicit euler, $c={c}, h=0.1$")
    plt.plot(xs, u_exact(xs, 1), label="analytic solution")
    plt.plot(nodes, u[:, -1], label="implicit euler")
    plt.xlabel("x")
    plt.ylabel("u(x)")
    plt.legend()
    plt.grid()
    plt.show()

# error analysis
errors = []
hs = [0.1, 0.01, 0.005]
for h in hs:
    c = 0.5
    k = c * h
    nodes = np.arange(0, 1 + h, h)
    ts = np.arange(0, 1 + k, k)

    u = implicit_euler(nodes, ts, h, k)

    error = np.sqrt(h) * np.linalg.norm(u_exact(nodes, 1) - u[:, -1])
    errors.append(error)

plt.figure()
plt.loglog(hs, errors, label="error")
plt.loglog(hs, hs, label="O(h)")
plt.xlabel('h')
plt.ylabel('error')
plt.grid()
plt.legend()
plt.show()
