import numpy as np
import matplotlib.pyplot as plt


def f(x):
    E = 0.01
    return np.array([2*x[0]+x[1]-E*(x[0]-x[1])**2,
                     x[0]+2*x[1]-3])


def df(x):
    E = 0.01
    return np.array([[2-2*E*(x[0]-x[1]), 1+2*E*(x[0]-x[1])],
                     [1, 2]])


def newton(x, f, df):
    d = np.linalg.solve(df(x),f(x))
    return x - d, np.linalg.norm(d)

A = np.array([[2,1],[1,2]])
b = np.array([0,3])
E = 0.01
x0 = np.linalg.solve(A,b)
steps = [i for i in range(12)]
errors = []
for i in steps:
    x0, error = newton(x0,f,df)
    errors.append(error)

plt.semilogy(np.array(steps)+1, errors, label="newton")
plt.xlabel("steps")
plt.ylabel("error")


def fn(x):
    return np.array([(x[0]-x[1])**2, 0])

def method(x):
    xn = np.linalg.solve(A,b+E*fn(x))
    return xn, np.linalg.norm(x0-xn)

x0_m = np.linalg.solve(A,b)
errors_m = []
for i in steps:
    x0_m, error = method(x0_m)
    errors_m.append(error)

plt.semilogy(np.array(steps)+1, errors_m, label="method")
plt.legend()
plt.xticks(np.array(steps)+1)
plt.show()