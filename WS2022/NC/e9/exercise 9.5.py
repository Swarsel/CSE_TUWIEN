import numpy as np
import matplotlib.pyplot as plt

def f(x):
    return np.arctan(x)

def H(x,s):
    return f(x) - (1-s)*np.arctan(4)

def df(x):
    return 1/(x**2+1)

def newton(x,s,H,df):
    return x - H(x,s)/df(x)

si = [i/10 for i in range(11)]
x0 = 4
x = [x0]

for s in si:
    for i in range(3):
        x0 = newton(x0,s,H,df)
        x.append(x0)

print(x0)
plt.semilogy([i for i in range(len(x))], np.abs(np.array(x)))
plt.xlabel("steps")
plt.ylabel("error")
plt.show()