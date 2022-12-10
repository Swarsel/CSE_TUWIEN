import numpy as np


def Cholesky(A):
    if not type(A) == np.ndarray:
        A = np.array(A)
    n = len(A)
    C = np.zeros(A.shape)
    # main routine
    for i in range(n):
        for k in range(i + 1):
            matmul = sum(C[i][j] * C[k][j] for j in range(k))
            # diagonal elements
            if (i == k):
                C[i][k] = np.sqrt(A[i][i] - matmul)
            else:
                C[i][k] = (1.0 / C[k][k] * (A[i][k] - matmul))
    return C


# create random symmetric, positive definite matrix of size n
n = 4
L = np.tril(np.random.rand(n))
A = L @ L.T

# print out input matrix A to be decomposed
print(f"Matrix to be decomposed is:\n{A}\n")

# decompose A
C = Cholesky(A)

# print out resulting lower triangular matrix
print(f"C=\n{C}\n")

# recompose A via C*C^T
A_composed = C @ C.T

# print recomposed A
print(f"A recomposed:\n{A_composed}\n")
print(f"Difference:\n{abs(A_composed - A)}\n")
