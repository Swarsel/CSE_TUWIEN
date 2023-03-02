import matplotlib.pyplot as plt
import numpy as np


def neville(x,f,x_eval):
    if len(x) != len(f):
        print(f"Warning: {x} and {f} do not have the same dimensions!")
    n = len(x)
    p = np.zeros((n, n))

    # populate first column
    p[:, 0] = f
    # iterate first over columns, so that we only use defined values
    for m in range(1,n):
        # columns get smaller by 1 each iteration
        for j in range(0,n-m):
            p[j][m] = ((x_eval - x[j]) * p[j+1][m-1] - (x_eval - x[j+m]) * p[j][m-1])/(x[j+m] - x[j])

    return p


def f(x):
    return (4 - x ** 2) ** (-1)


def T(n):
    return (1 / 12) * (4 - 4 ** (-n))


ns = [i for i in range(1, 21)]
xs = np.linspace(-1, 1, 100)
f_exact = f(xs)
max_errors = []
for n in ns:
    f_approx = []
    f_error = []
    x_cheb = [np.cos(np.pi * (2* i + 1) / (2* n + 2)) for i in range(n + 1)]
    f_cheb = [f(x) for x in x_cheb]
    for x in xs:
        approx = neville(x_cheb, f_cheb, x)[0][-1]
        f_approx.append(approx)
        f_error.append(abs(f(x) - approx))
    max_errors.append(max(f_error))

plt.title("Error of Chebyshev and Taylor approximations at different n")
plt.semilogy(ns, [f(1) - T(n) for n in ns], label="Taylor")
plt.semilogy(ns, max_errors, label="Chebyshev")
plt.ylabel("error")
plt.xlabel("n")
plt.legend()
plt.show()