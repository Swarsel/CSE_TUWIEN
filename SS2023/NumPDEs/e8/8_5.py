import numpy as np
import scipy as sp
from scipy.integrate import quad
import matplotlib.pyplot as plt

def fem_1d(nodes):
    # equidistant grid points
    N = len(nodes) -1
    # vector of elements dependant on nodes
    dx = nodes[1:] - nodes[:-1]

    # stiffness matrix
    A = np.diag(1 / dx[:-1] + 1 / dx[1:])
    A = A - np.diag(1 / dx[1:-1], 1)
    A = A - np.diag(1 / dx[1:-1], -1)
    A = sp.sparse.csc_matrix(A)

    # load vector
    def left(x, a, b):
        #return (2 / 9 * (x ** (-4 / 3) + 5 * x ** (-1 / 3))) * ((x - a) / (b - a))
        return ((21*x + 3)/(16 * x ** (5/4))) * ((x - a) / (b - a))

    def right(x, a, b):
        #return (2 / 9 * (x ** (-4 / 3) + 5 * x ** (-1 / 3))) * ((b - x) / (b - a))
        return ((21*x + 3)/(16 * x ** (5/4))) * ((b - x) / (b - a))

    f1 = np.zeros(N - 1)
    f2 = np.zeros(N - 1)
    for i in range(1, N):
        f1[i - 1], _ = quad(lambda x: left(x, x_vals[i - 1], x_vals[i]), x_vals[i - 1], x_vals[i])
        f2[i - 1], _ = quad(lambda x: right(x, x_vals[i], x_vals[i + 1]), x_vals[i], x_vals[i + 1])
    f = (f1 + f2).T

    u = sp.sparse.linalg.spsolve(A, f)
    u = np.concatenate(([0], u.flatten(), [0]))  # add boundaries

    return A, f, u

def energy_norm(u, f):
    def du_l2(x):
        return (2 / 3 * x ** (-1 / 3) - 5 / 3 * x ** (2 / 3)) ** 2

    norm_u, _ = quad(lambda x: du_l2(x), 0, 1)
    norm_uN = u[1:-1].T @ f
    #error = norm_u - norm_uN
    error = 3/5 - norm_uN
    return error


def estimate(u, nodes):
    def f_squared(x):
        #return (2 / 9 * (x ** (-4 / 3) + 5 * x ** (-1 / 3))) ** 2
        return (9*(7*x+1) ** 2) / (256*x ** (5/2))

    dx = nodes[1:] - nodes[:-1]
    midpoints = 0.5 * (nodes[1:] + nodes[:-1])

    f_mid = f_squared(midpoints)
    f_norm = dx * f_mid

    u_diff = u[1:] - u[:-1]
    u_diff /= dx

    normal_jump = (u_diff[1:] - u_diff[:-1]) ** 2
    normal_jump = np.concatenate(([0], normal_jump.flatten(), [0]))  # add boundaries
    normal_jump = normal_jump[1:] + normal_jump[:-1]
    res = dx ** 2 * f_norm + dx * normal_jump
    return res
    
# step size
h = 0.2
# equidistant grid points
x_vals = np.arange(0, 1 + h, h)

# marking parameter
percent = 1
# tolerance for eta - stopping criterion
tol = 0.008
# set initial eta to be bigger than tolerance
eta= tol + 1
eta_sum = tol + 1


# list of number of elements
no_elements = [0]
# list of errors
errors = [0]
# list of etas
etas = [0]
# loop counter
k = 1

while eta_sum > tol:
    ## SOLVE

    # solve poisson problem
    A, f, u = fem_1d(x_vals)
     
    ## ESTIMATE
    
    # local error contributions
    eta_T = estimate(u, x_vals)
    # cumulative sum of local error estimators
    idx = np.argsort(eta_T)[::-1]
    eta = np.cumsum(eta_T[idx])
    eta_sum = eta[-1]
    
    ## MARK
    
    # indices of selected elements
    marked = []
    for i in range(len(eta_T)):
        print(eta_T[i])
        if eta_T[i] >= percent * eta_sum:
            marked.append(i)
    #marked = idx[:np.argmax(eta >= percent * eta_sum)+1 ]
    marked_sorted = np.sort(marked)
    print(marked_sorted)
    
    ## REFINE

    for i in marked_sorted:
        x_vals = np.concatenate((x_vals[:marked_sorted[i]+i+1],
                                 0.5* (x_vals[marked_sorted[i]+i] + x_vals[marked_sorted[i]+i+1]),
                                 x_vals[marked_sorted[i]+i+1:]), axis=None)

    # store current number of elements
    no_elements.append(x_vals.size)
    # store current error
    errors.append(energy_norm(u, f))
    # store current eta
    etas.append(eta_sum)
    # increment loop counter
    k = k + 1




