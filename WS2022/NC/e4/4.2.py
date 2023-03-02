import numpy as np
import matplotlib.pyplot as plt

def f1(x):
    return np.e ** x

def f2(x):
    return np.e ** (-1/x)

def f3(x):
    return np.e ** (-1/(x**2))

xs = list(range(100))
f1y = [f1(i) for i in xs]
f2y = [f1(i) for i in xs]
f3y = [f1(i) for i in xs]

#plt.semilogy(xs, f1y)
#plt.semilogy(xs, f2y)
plt.semilogx(xs, f3y)
plt.show()