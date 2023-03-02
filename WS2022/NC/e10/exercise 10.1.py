import matplotlib.pyplot as plt
import numpy as np


def f(u, i):
    h = 1/len(u)
    if i == len(u)-1:
        return (2 * u[i] - u[i - 1]) / h ** 2 + u[i] ** 3 - 1
    else:
        return (-u[i+1]+2*u[i]-u[i-1])/h**2 + u[i]**3 - 1


def df(u, i):
    h = 1 / len(u)
    # if i == 1:
    #     print(i,u[i+1],u[i])
    #     return (f(u, i + 1) - f(u, i)) / (u[i + 1] - u[i])
    # if i == len(u)-2:
    #     return (f(u, i) - f(u, i - 1)) / (u[i] - u[i - 1])
    # else:
    # print(i, u[i + 1], u[i-1])
    return (f(u, i+1) - f(u, i-1))/(2*h)


def newton(u):
    for i in range(1, len(u)-1):
        u[i] = u[i] - f(u, i)/df(u, i)
    return u


N = 10
u = [1 for i in range(N)]
# u = np.random.rand(N)
u[0] = 0
u[-1] = 0
it = 10
us = np.zeros((it+1,N))
us[0] = 0
its = [i for i in range(it)]
for i in its:
    u = newton(u)
    us[i+1]=u

errors = []
for i in range(it):
    errors.append(np.linalg.norm(us[i]-us[i+1]))

print(errors)
plt.semilogy(np.array(its)+1, errors)
plt.show()
