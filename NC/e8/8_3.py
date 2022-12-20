import numpy as np
import matplotlib.pyplot as plt


def N(n):
    return 2 ** n


# set range of n
n_min, n_max = 2, 10
n_vals = list(range(n_min, n_max + 1))

# Taylor series solution for a1 and a3
a1 = 1
a3 = -1 / 6

a1_error = []
a3_error = []

# set up A and b matrices of Ax = b
A = np.zeros((n_max - n_min + 1, 2))
b = np.zeros((n_max - n_min + 1, 1))

for n in n_vals:
    # create random points from given interval
    x_j = np.random.uniform(-1 / N(n), 1 / N(n), 1)

    # populate matrix A
    A[n - 2][0] = x_j
    A[n - 2][1] = x_j ** 3

    # populate vector b
    b[n - 2] = np.sin(x_j)

    # Fulfill (A^T * A) * x = A^T * b -> x = (A^T * A)^(-1) * (A^T * b)
    # A^T
    A_T = A.T

    # A^T * A
    A_T_A = A_T @ A

    # (A^T * A)^(-1)
    A_T_A_inv = np.linalg.inv(A_T_A)

    # (A^T * b)
    A_T_b = A_T @ b

    # x = (A^T * A)^(-1)(A^T * b)
    x = A_T_A_inv @ A_T_b

    a1_error.append(abs(x[0][0] - a1))
    a3_error.append(abs(x[1][0] - a3))

plt.semilogy([n for n in n_vals], a1_error, label="a1")
plt.semilogy([n for n in n_vals], a3_error, label="a3")
plt.xlabel("n")
plt.ylabel("error")
plt.legend()
plt.show()
