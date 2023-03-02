import numpy as np
import sympy as sym
import matplotlib.pyplot as plt


def simpson(f, a, b, N=1):
    S = 0
    for i in range(N):
        x1 = a + (b - a) * i / N
        x2 = a + (b - a) * (i + 1) / N
        hi = x2 - x1

        S += (f(x1) + 4 * f((x1 + x2) / 2) + f(x2)) * hi / 6
    return S


def func(x):
    if x < (1 / 3):
        return (1 / 2) * np.e ** x
    else:
        return np.e ** x

def adapt(f,a,b,hmin, t):
    s = simpson(f,a,b)
    if b - a <= hmin:
        return s
    m = (a+b)/2
    if abs(simpson(f,a,m) + simpson(f,m,b) - s) <= t:
        return s
    else:
        I = adapt(f,a,m,hmin,t/2) + adapt(f,m,b,hmin,t/2)
        return I


tau = [2 ** (-i) for i in range(11)]
error = []
x = sym.symbols("x")
for i in range(11):
    exact = sym.integrate((1/2) * (np.e ** x), (x,0,1/3)) \
            + sym.integrate(np.e ** x, (x,1/3,1))
    error.append(abs(adapt(func,0,1,tau[i],tau[i]) - exact))

plt.title("error of adapt algorithm")
plt.xlabel("tau")
plt.ylabel("error")
plt.loglog(tau,error, label="error")
plt.loglog(tau,tau,linestyle="--",  label="tau")
plt.legend()
plt.show()