import numpy as np
from sklearn.datasets import make_spd_matrix

def Cholesky(A):
    if not type(A) == np.ndarray:
        A = np.array(A)
    n = len(A)
    C = np.zeros(A.shape)
    for i in range(n):
        for k in range(i + 1):
            matmul = sum(C[i][j] * C[k][j] for j in range(k))
            # diagonal elements
            if (i == k):
                C[i][k] = np.sqrt(A[i][i] - matmul)
            else:
                C[i][k] = (A[i][k] - matmul) / C[k][k]
    return C

def CholeskyCCT(A):
    if not type(A) == np.ndarray:
        A = np.array(A)
    n = len(A)
    C = np.zeros(A.shape)
    for i in range(n):
        for k in range(i + 1):
            matmul = sum(C[i][j] * C[k][j] for j in range(k))
            # diagonal elements
            if (i == k):
                C[i][k] = np.sqrt(A[i][i] - matmul)
            else:
                C[i][k] = (A[i][k] - matmul) / C[k][k]
    return C


# create random symmetric, positive definite matrix of size n
n = 4
A = make_spd_matrix(n)

# (alternative for lower triagonal A)
# L = np.tril(np.random.rand(n))
# A = L @ L.T

print(f"Matrix to be decomposed is:\n{A}\n")

# decompose A
C = Cholesky(A)
print(f"C=\n{C}\n")

A_composed = C @ C.T
print(f"A recomposed:\n{A_composed}\n")
print(f"Difference:\n{abs(A_composed - A)}\n")
