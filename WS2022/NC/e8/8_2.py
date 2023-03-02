import numpy as np

# set up A and b matrices of Ax = b
A = np.array([[1, 0, 0],
              [0, 1, 0],
              [0, 0, 1],
              [0, 2, 2],
              [2, 0, 2]])
b = np.array([[26], [38], [55], [188], [163]])

# Fulfill (A^T * A) * x = A^T * b -> x = (A^T * A)^(-1) * (A^T * b)
# A^T
A_T = A.T

# A^T * A
A_TA = A_T @ A

# (A^T * A)^(-1)
A_TA_inv = np.linalg.inv(A_TA)

# (A^T * b)
A_Tb = A_T @ b

# x = (A^T * A)^(-1) * (A^T * b)
x = A_TA_inv @ A_Tb

print(f"edge 1: {round(x[0][0], 0)} mm\nedge 2: {round(x[1][0], 0)} mm\nedge 3: {round(x[2][0], 0)} mm")
