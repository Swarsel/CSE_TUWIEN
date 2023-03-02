import numpy as np
import matplotlib.pyplot as plt


def composite_gauss(n, L, q, f, Ls=1):
    x, w = np.polynomial.legendre.leggauss(n)
    # get intervals
    subintervals = [0]
    for i in range(1, L + 1):
        subintervals.append(Ls * q ** (L - i))
    Q = 0
    for i in range(L):
        a = subintervals[i]
        b = subintervals[i + 1]
        # [-1, 1] -> [a, b]
        x_scaled = 0.5 * (x + 1) * (b - a) + a
        w_scaled = 0.5 * w * (b - a)
        for i in range(len(x_scaled)):
            Q += w_scaled[i] * f(x_scaled[i])

    return Q


def func1(x):
    # acquired by substitution u=1/x
    return np.log(1 / x) * x ** (np.pi - 2)


def func2(x):
    return x * np.exp(x * (1 - np.pi))


print(composite_gauss(20, 20, 0.15, func1))
print(composite_gauss(20, 20, 0.15, func2, Ls=20))

analytical = 1 / (np.pi ** 2 - 2 * np.pi + 1)

ns = list(range(1, 21))
error1 = []
error2 = []
for n in ns:
    error1.append(abs(analytical - composite_gauss(n, n, 0.15, func1)))
    error2.append(abs(analytical - composite_gauss(n, n, 0.15, func2, Ls=n)))

plt.semilogy(ns, error1, label="$\log(\\frac{1}{x}) x^{\pi-2}$")
plt.semilogy(ns, error2, label="$x e^{x (1 - \pi)}$")
plt.title("Composite Gauss error")
plt.xlabel("n")
plt.ylabel("error")
plt.legend()
plt.show()
