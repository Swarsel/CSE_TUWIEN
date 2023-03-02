import numpy as np
import matplotlib.pyplot as plt

def power_iteration(A, num_iterations: int):
    b_k = np.random.rand(A.shape[1])
    b_k = b_k / np.linalg.norm(b_k)
    l = [np.dot(np.conj(b_k),np.dot(A,b_k))]
    for _ in range(num_iterations):
        b_k1 = np.dot(A, b_k)
        b_k1_norm = np.linalg.norm(b_k1)
        b_k = b_k1 / b_k1_norm
        l.append(np.dot(np.conj(b_k),np.dot(A,b_k)))

    return b_k, l


def inverse_iteration(A, num_iterations: int):
    b_k = np.random.rand(A.shape[1])
    b_k = b_k / np.linalg.norm(b_k)
    A = np.linalg.inv(A)
    l = [1/np.dot(np.conj(b_k), np.dot(A, b_k))]
    for _ in range(num_iterations):
        b_k1 = np.dot(A, b_k)
        b_k1_norm = np.linalg.norm(b_k1)
        b_k = b_k1 / b_k1_norm
        l.append(1/np.dot(np.conj(b_k),np.dot(A,b_k)))

    return b_k, l

A = np.array([[1, 5],
              [5, 6]])
its = 15
eigvals = np.linalg.eigvals(A)
print(eigvals)
print((-5*5**0.5+7)/2)
print((5*5**0.5+7)/2)
x1, l1 = power_iteration(A, its)
x2, l2 = inverse_iteration(A, its)
print(l2)
plt.semilogy()
plt.plot(range(its+1), np.abs(np.array(l1)-np.amax(eigvals)), label="power")
plt.plot(range(its+1), np.abs(np.array(l2)-np.amin(eigvals)), label="inverse")
plt.legend()
plt.xlabel("iterations")
plt.ylabel("error")
plt.show()
