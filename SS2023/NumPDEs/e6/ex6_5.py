import numpy as np
import matplotlib.pyplot as plt


def evalLegendre(x, k):
    if k == 1:
        return x
    if k == 0:
        return 1
    else:
        return ((2 * k - 1) * x * evalLegendre(x, k - 1) - (k - 1) * evalLegendre(x, k - 2))/k


def eval_integrated_legendre(x, k):
    return (evalLegendre(x, k + 1) - evalLegendre(x, k - 1)) / (2 * k + 1)


def gauss(f, m, k):
    x, w = np.polynomial.legendre.leggauss(m)
    return sum(w * f(x) * eval_integrated_legendre(x, k))


x = np.linspace(-1, 1, 100)
for k in range(0, 5):
    plt.plot(x, [evalLegendre(xi, k) for xi in x], label=f"k={k}")
plt.title("Legendre Polynomials by recursion formula")
plt.legend()
plt.ylim(-1.5,1.5)
plt.grid()
plt.show()

for k in range(1, 5):
    plt.plot(x, [eval_integrated_legendre(xi, k) for xi in x], label=f"k={k}")
plt.title("Integrated Legendre Polynomials")
plt.legend()
plt.ylim(-1.5,1.5)
plt.grid()
plt.show()


def sinus(x):
    return np.sin(x)


def cosinus(x):
    return np.cos(x)


def poly(x):
    return x ** 3


for k in range(2, 5):
    for m in [5, 10, 15, 20]:
        print(f"k={k}, m={m}, cos: {gauss(cosinus, m, k)}")
        print(f"k={k}, m={m}, sin: {gauss(sinus, m, k)}")
        print(f"k={k}, m={m}, x^3: {gauss(poly, m, k)}")
