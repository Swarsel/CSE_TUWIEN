import numpy as np
import matplotlib.pyplot as plt


def f(x):
    return np.exp(-x ** 2) * np.sin(x) ** 2


def Q(f, N):
    q = 0
    h = 1 / np.sqrt(N)
    for i in range(-N, N + 1):
        xi = i * h
        q += h * f(xi)
    return q


analytic = ((np.e - 1) * np.sqrt(np.pi)) / (2 * np.e)
ns = list(range(1, 100))

errors = []
for N in ns:
    errors.append(abs(Q(f, N) - analytic))

plt.title("Quadrature error of trapezoidal rule")
plt.semilogy(ns, errors)
plt.xlabel("N")
plt.ylabel("error")
plt.show()
