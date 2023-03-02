import matplotlib.pyplot as plt
import numpy as np
import functions

# a)
ns = [10, 20, 40]
interval = (-5, 5)


def neville(x, f, x_eval):
    if len(x) != len(f):
        print(f"Warning: {x} and {f} do not have the same dimensions!")
    n = len(x)
    p = np.zeros((n, n))

    # populate first column
    p[:, 0] = f
    # iterate first over columns, so that we only use defined values
    for m in range(1, n):
        # columns get smaller by 1 each iteration
        for j in range(0, n - m):
            p[j][m] = ((x_eval - x[j]) * p[j + 1][m - 1] - (x_eval - x[j + m]) * p[j][m - 1]) / (x[j + m] - x[j])

    return p


def f(x):
    return (1 + x ** 2) ** (-1)


x_exact = np.linspace(interval[0], interval[1], 100)
f_exact = f(x_exact)

lebesque_c_unif = []
lebesque_c_cheb = []
for n in ns:
    f_approx_unif = []
    f_approx_cheb = []
    x_unif = [5 * (-1 + 2 * (i / n)) for i in range(n + 1)]
    x_cheb = [5 * np.cos(np.pi * ((i + 0.5) / (n + 1))) for i in range(n + 1)]
    f_unif = [f(x) for x in x_unif]
    f_cheb = [f(x) for x in x_cheb]
    for x in x_exact:
        f_approx_unif.append(neville(x_unif, f_unif, x)[0][-1])
        f_approx_cheb.append(neville(x_cheb, f_cheb, x)[0][-1])

    plt.title(f"Approximations for n={n}")
    plt.plot(x_exact, f_exact, label="Analytical")
    plt.plot(x_exact, f_approx_unif, label="Uniform")
    plt.plot(x_exact, f_approx_cheb, label="Chebyshev")
    plt.ylim(-0.5, 1.2)
    plt.xlabel("x")
    plt.ylabel("f(x)")
    plt.legend()
    plt.show()

    # b)
    lebesque_c_unif.append(functions.lebesgue_const(x_exact, x_unif))
    lebesque_c_cheb.append(functions.lebesgue_const(x_exact, x_cheb))

#c
coefs = np.polyfit([20,40], np.log(lebesque_c_unif[1:]), deg=1)
b = coefs[0]
C = np.exp(coefs[1])

plt.title("Lebesgue constant of uniform and chebishev points")
plt.semilogy(ns, lebesque_c_unif, label="Uniform")
plt.semilogy(ns, lebesque_c_cheb, label="Chebyshev")
plt.plot(ns, [C*np.exp(b*n) for n in ns],
         ls="dashed", label=r"$C\cdot e^{bn}$")
plt.ylabel("$\Lambda_n$")
plt.xlabel("n")
plt.legend()
plt.show()