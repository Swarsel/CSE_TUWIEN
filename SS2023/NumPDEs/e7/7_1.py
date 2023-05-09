import numpy as np
import matplotlib.pyplot as plt


def exact(x):
    """ exact solution for -u'' + u = 2sin(x) """
    return np.sin(x)


def nodes(n, a=0, b=np.pi):
    """ shorthand for getting the nodes for mesh fineness N """
    return np.linspace(a, b, n)


def leg(x, k):
    """ returns i-th legendre polynomial at x using three-term-recurrence relation"""
    if k == 1:
        return x
    if k == 0:
        return 1.0
    else:
        return ((2 * k - 1) * x * leg(x, k - 1) - (k - 1) * leg(x, k - 2)) / k


def int_leg(x, k):
    """ returns i-th integrated legendre polynomial at x """
    # we redefine N0 and N1 here according to the script p.44 (there called N1 and N2 respectively)
    if k == 0:
        return x
    if k == 1:
        return 1 - x
    else:
        return (leg(x, k + 1) - leg(x, k - 1)) / (2 * k + 1)


def gauss_a(p, i, j, a=-1, b=1):
    """
    solves lhs a(phi1, phi2) = int (nabla phi1 nabla phi2 + phi1 * phi2)
    using the integrated legendres as base
    """
    x, w = np.polynomial.legendre.leggauss(p)
    # basis functions are the Ni -> use Li for nabla phi
    x = ((b - a) / 2) * x + ((b + a) / 2)
    return ((b - a) / 2) * sum(w * (leg(x, i) * leg(x, j) + int_leg(x,i) * int_leg(x, j)))


def gauss_f(p, i, a=-1, b=1):
    """ solves rhs l(phi1) = int( 2sin(x) * phi1) using the integrated legendres as base"""
    x, w = np.polynomial.legendre.leggauss(p)
    x = ((b - a) / 2) * x + ((b + a) / 2)
    return ((b - a) / 2) * sum(w * (2 * np.sin(x) * int_leg(x, i)))


def connectivity_T(N, p, i):
    """
    returns the connectivity matrix for the element T
    naming scheme for nodal functions (case p=3, N=4 here)
    old:
    0 - 4 - 7 - 1 - 5 - 8 - 2 - 6 - 9 - 3
    revised:
    0 - 1 - 2 - 3 - 4 - 5 - 6 - 7 - 8 - 9
    """
    dim = p + 1
    # num_elem = len(N) - 1
    num_global_functions = calc_num_global_functions(p, N)
    CT = np.zeros((dim, num_global_functions))
    # nodes
    # old:
    # CT[0][i] = 1.0
    # CT[1][i + p] = 1.0

    # edges

    # old:
    # for j in range(2, dim)
    for j in range(dim):
        # each element gets an edge function -> every num_elementh' function belongs to it
        # we are starting at j - 1 because we already have the first two ones
        # also shift by 1 to get away from the nodes

        # old:
        # CT[j][i + 1 + p * (j - 1)] = 1.0
        CT[j][p * i + j] = 1.0

    CT = CT.T
    return CT


def connectivity(N, p):
    """ returns a list of all connectivity matrices """
    num_elem = len(N) - 1
    return [connectivity_T(N, p, t) for t in range(num_elem)]


def element_stiffness_matrix(p, N):
    """ returns a list of all element stiffness matrices"""
    dim = p + 1
    num_elem = len(N) - 1

    AT = []
    for t in range(num_elem):
        A_T = np.zeros((dim, dim))
        for i in range(dim):
            for j in range(dim):
                A_T[i][j] = gauss_a(p, i, j, a=N[t], b=N[t + 1])
        AT.append(A_T)
    return AT


def element_load_vector(p, N):
    """ returns a list of all element load vectors """
    dim = p + 1
    num_elem = len(N) - 1

    fT = []
    for t in range(num_elem):
        f_T = np.zeros((dim, 1))
        for i in range(dim):
            f_T[i][0] = gauss_f(p, i, a=N[t], b=N[t + 1])
        fT.append(f_T)
    return fT


def global_stiffness_matrix(N, p, AT, CT):
    """ assembles the global stiffness matrix using sum(CAC.T) """
    num_global_functions = calc_num_global_functions(p, N)

    A = np.zeros((num_global_functions, num_global_functions))
    for a, c in zip(AT, CT):
        A += c @ a @ c.T
    return A


def global_load_vector(N, p, fT, CT):
    """ assmbles the global load vector using sum(Cf) """
    num_global_functions = calc_num_global_functions(p, N)

    F = np.zeros((num_global_functions, 1))
    for f, c in zip(fT, CT):
        F += c @ f
    return F


def energy_norm(x, A):
    """ calculates the energy norm :) """
    return np.sqrt((x.T @ A @ x)[0][0])

def calc_num_global_functions(p, N):
    """ returns the total number of nodal functions """
    num_elem = len(N) - 1
    num_edge_functions = (p - 1) * num_elem
    num_node_functions = len(N)
    return num_node_functions + num_edge_functions

def plot():
    """ plotting the problem for p(3->5) and N(4->10) """
    # plot for different p and N
    for p in range(3, 5):
        error_energy_norm = []
        N_list = []

        for n in range(4, 10):
            print(f"Working on p={p},N={n}")
            N = nodes(n)
            num_global_functions = calc_num_global_functions(p, N)
            N_split = np.linspace(0, np.pi, num_global_functions)
            fT = element_load_vector(p, N)
            AT = element_stiffness_matrix(p, N)
            CT = connectivity(N, p)
            A = global_stiffness_matrix(N, p, AT, CT)
            F = global_load_vector(N, p, fT, CT)

            # remove dirichlet entries
            A = A[1:-1, 1:-1]
            F = F[1:-1]
            N_split = N_split[1:-1]

            u = np.linalg.solve(A, F)
            error = np.array([abs(ui - exact(n)) for ui, n in zip(u, N_split)])
            N_list.append(n)
            error_energy_norm.append(energy_norm(error, A))

        plt.plot([n for n in N_list], [error for error in error_energy_norm])
        plt.title(f"Energy norm error for p={p}")
        plt.xlabel("N")
        plt.ylabel("error")
        plt.yscale("log")
        plt.show()


if __name__ == "__main__":
    # plot()

    # testbench
    p = 3  # order
    N = np.linspace(0, np.pi, 4)
    num_global_functions = calc_num_global_functions(p, N)
    N_split = np.linspace(0, np.pi, num_global_functions)
    lb, rb = 0, np.pi
    fT = element_load_vector(p, N)
    AT = element_stiffness_matrix(p, N)
    CT = connectivity(N, p)
    A = global_stiffness_matrix(N, p, AT, CT)
    F = global_load_vector(N, p, fT, CT)

    # remove dirichlet entries
    A = A[1:-1, 1:-1]
    F = F[1:-1]
    N_split = N_split[1:-1]

    u = np.linalg.solve(A, F)
    error = np.array([abs(ui - exact(n)) for ui, n in zip(u, N_split)])
    normed = energy_norm(error, A)
    print(energy_norm(error, A))




