import numpy as np
import sympy as sym
import matplotlib.pyplot as plt


def midpoint(f, a, b, c, d):
    fm = f((a + b) / 2, (c + d) / 2)
    A = (b - a) * (d - c)
    Q = A * fm
    return Q


def func1(x, y):
    return x ** 2


def func2(x, y):
    if x < y:
        return 0
    else:
        return 1


def adapt(f, a, b, c, d, hmin, t):
    midpoint_result = midpoint(f, a, b, c, d)
    if b - a <= hmin:
        return midpoint_result
    mx = (a + b) / 2
    my = (c + d) / 2
    midpoint_acc = midpoint(f, a, mx, c, my) + midpoint(f, a, mx, my, d) \
                   + midpoint(f, mx, b, c, my) + midpoint(f, mx, b, my, d)
    if abs(midpoint_acc - midpoint_result) <= t:
        return midpoint_result
    else:
        I = adapt(f, a, mx, c, my, hmin, t / 4) + adapt(f, a, mx, my, d, hmin, t / 4) \
            + adapt(f, mx, b, c, my, hmin, t / 4) + adapt(f, mx, b, my, d, hmin, t / 4)
        return I


tau = [2 ** (-i) for i in range(16)]
error1 = []
error2 = []
for i in range(16):
    error1.append(abs(adapt(func1, 0, 1, 0, 1, tau[i], tau[i]) - 1/3))
    error2.append(abs(adapt(func2, 0, 1, 0, 1, tau[i], tau[i]) - 1/2))

plt.title("error of adapt algorithm")
plt.xlabel("tau")
plt.ylabel("error")
plt.loglog(tau, error1, label="error1")
plt.loglog(tau, error2, label="error2")
plt.loglog(tau, tau, linestyle="--", label="tau")
plt.legend()
plt.show()
