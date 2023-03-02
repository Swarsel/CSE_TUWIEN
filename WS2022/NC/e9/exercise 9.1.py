import numpy as np
import matplotlib.pyplot as plt

def f1(x):
    return x**2

def df1(x):
    return 2*x

def f2(x):
    return np.exp(x)-2

def df2(x):
    return np.exp(x)

def f3(x):
    return abs(x)**(3/2)

def df3(x):
    return 3/2 * x/np.sqrt(abs(x))

def f4(x):
    return 1/x - 1

def df4(x):
    return -1/x**2

def newton(x,f,df):
    return x - f(x)/df(x)

x_start = [0.5,0.5,0.5,2.1]
x_node = [0, np.log(2), 0, 1]

for k,f in enumerate([(f1,df1),(f2,df2),(f3,df3),(f4,df4)]):
    steps = [i for i in range(1,6)]
    x0 = x_start[k]
    x = [x0]
    for i in steps:
        x0 = newton(x0,f[0],f[1])
        x.append(x0)
    plt.semilogy([0]+steps, np.abs(np.array(x) - x_node[k]), label="f"+str(k+1))
    plt.semilogy(steps, np.abs(np.array(x[:-1]) - x_node[k])**2, "--", color="black", label="f" + str(k + 1) + "quadratic")


plt.legend()
plt.xlabel("steps")
plt.ylabel("error")
plt.show()