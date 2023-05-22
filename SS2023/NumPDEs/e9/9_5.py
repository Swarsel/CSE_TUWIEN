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


def explicit_euler(nodes, ts, h, k):
    u_0 = u_exact(nodes, 0)
    A = stiffness_matrix(nodes)
    M = mass_matrix(h)
    u = np.zeros((len(nodes), len(ts)))
    u[:, 0] = u_0
    col = 0
    for t in ts[:-1]:
        u[1:-1, col + 1] = u[1:-1, col] + k * (sp.sparse.linalg.spsolve(M, load_vector(nodes, t)) - sp.sparse.linalg.spsolve(M, A * u[1:-1,col]))
        col += 1
    return u


def u_exact(x, t):
    return np.exp(-t) * x * (1 - x)

# instability
for h in [0.1, 0.05]:
    for n in [1.99, 2.01]:
        nodes = np.arange(0, 1 + h, h)
        A = stiffness_matrix(nodes)
        M = mass_matrix(h)
        max_eig = max(sp.sparse.linalg.eigs(sp.sparse.linalg.spsolve(M, A), k=1, which='LM')[0].real)
        k = n / max_eig
        ts = np.arange(0, 1 + k, k)

        u = explicit_euler(nodes, ts, h, k)
        xs = np.linspace(0, 1, 100)
        plt.title(f"exact vs explicit euler, $n={n}, h={h}$")
        plt.plot(xs, u_exact(xs, 1), label="analytic solution")
        plt.plot(nodes, u[:, -1], label="explicit euler")
        plt.xlabel("x")
        plt.ylabel("u(x)")
        plt.legend()
        plt.grid()
        plt.show()

# error analysis
hs = [0.25, 0.1, 0.05]
for n in [2.01, 1.99]:
    errors = []
    evs = []
    for h in hs:
        nodes = np.arange(0, 1 + h, h)
        A = stiffness_matrix(nodes)
        M = mass_matrix(h)
        max_eig = max(sp.sparse.linalg.eigs(sp.sparse.linalg.spsolve(M,A), k=1, which='LM')[0].real)
        k = n/max_eig
        ts = np.arange(0, 1 + k, k)

        u = explicit_euler(nodes, ts, h, k)

        error = np.sqrt(h) * np.linalg.norm(u_exact(nodes, 1) - u[:, -1])
        evs.append(max_eig)
        errors.append(error)

    plt.title(f"Explicit euler - $n={n}, h={h}$ errors")
    plt.loglog(hs, errors, label="error")
    plt.loglog(hs, hs, label="O(h)")
    plt.xlabel('h')
    plt.ylabel('error')
    plt.grid()
    plt.legend()
    plt.show()

# max eigenvalues
plt.title("Max eigenvalues over h")
plt.loglog(hs, evs, '-o')
plt.xlabel('h')
plt.ylabel('$\\lambda_{max}$')
plt.grid()
plt.show()