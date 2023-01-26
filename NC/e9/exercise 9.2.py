import numpy as np
import matplotlib.pyplot as plt

def f(x):
    return np.array([3*x[0]-np.cos(x[1]*x[2])-3/2,
                     4*x[0]**2-625*x[1]**2+2*x[2]-1,
                     20*x[2]+np.exp(-x[0]*x[1])+9])


def df(x):
    return np.array([[3, x[2]*np.sin(x[1]*x[2]), x[1]*np.sin(x[1]*x[2])],
                     [8*x[0], -1250*x[1], 2],
                     [-x[1]*np.exp(-x[0]*x[1]),-x[0]*np.exp(-x[0]*x[1]), 20]])


# def newton(x, f, df):
#     return x - np.linalg.inv(df(x))@f(x)


def newton(x, f, df):
    d = np.linalg.solve(df(x),f(x))
    return x - d, np.linalg.norm(d)

x0 = np.array([1,1,1])
steps = [i for i in range(13)]
errors = []
for i in steps:
    x0, error = newton(x0,f,df)
    errors.append(error)

plt.semilogy(np.array(steps)+1, errors)
plt.xticks(np.array(steps)+1)
plt.xlabel("steps")
plt.ylabel("error")
plt.show()