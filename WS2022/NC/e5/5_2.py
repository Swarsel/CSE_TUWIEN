import numpy as np
import matplotlib.pyplot as plt


def composite_gauss(n, L, q, f):
    x, w = np.polynomial.legendre.leggauss(n)
    # get intervals
    subintervals = [0]
    for i in range(1, L + 1):
        subintervals.append(q ** (L - i))
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


for m in [0, 1, 2]:
    print(composite_gauss(20, 20, 1, lambda x: x ** m))


def func(x):
    return x ** 0.1 * np.log(x)


exact = -1 / (1.1 ** 2)

quadrature_error = []
qs = [0.5, 0.15, 0.05]
ns = list(range(1, 21))
color = ["r", "g", "b"]

for (q, c) in zip(qs, color):
    error_q = []
    for n in ns:
        error_q.append(abs(composite_gauss(n, n, q, func) - exact))
    quadrature_error.append(error_q)
    plt.semilogy(ns, error_q, color=c, label="q=" + str(q))
    poly = np.polyfit(ns, np.log(error_q), 1)
    plt.semilogy(ns, np.exp([np.polyval(poly, n) for n in ns]), linestyle="--", color=c, label="fit for q=" + str(q))

# q=0.15 is the best choice
plt.title("Composite Gauss error")
plt.xlabel("n")
plt.ylabel("error")
plt.legend()
plt.show()
