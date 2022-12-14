import numpy as np

# set up A and b matrices of Ax = b
A = np.array([ [1,0,0], [0,1,0], [0,0,1], [0,2,2], [2,0,2] ])
b = np.array([ [26], [38], [55], [188], [163] ])

# A^T
A_T = np.transpose(A)

# A^T * A
A_TA = np.matmul(A_T, A)

# (A^T * A)^(-1)
A_TA_inv = np.linalg.inv(A_TA)

# (A^T * b)
A_Tb = np.matmul(A_T, b)

# x = (A^T * A)^(-1)
x = np.matmul(A_TA_inv, A_Tb)

print(f"edge 1: {round(x[0][0], 4)} mm, edge 2: {round(x[1][0], 4)} mm, edge 3: {round(x[2][0], 4)} mm")