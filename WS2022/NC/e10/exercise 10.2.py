import numpy as np
import matplotlib.pyplot as plt

def F(x):
    return 2-x**2-np.exp(x)


def dF(x):
    return -2*x-np.exp(x)


def secant(xn,xn1,F,dF):
    if F(xn) == F(xn1):
        return xn - F(xn)/dF(xn)
    else:
        return xn - (xn-xn1)/(F(xn)-F(xn1)) * F(xn)

def Newton(x,F,dF):
    return x - F(x)/dF(x)


x0=2.5
x1=x0-F(x0)/dF(x0)
steps = 0
xs = [x1]
ps_newton=[]
while abs(x1-x0) > np.finfo(float).eps:
    x0 = x1
    x1 = Newton(x1,F,dF)
    xs.append(x1)
    steps += 1
x_sol = x1
print("Newton: ", "steps=",steps,"x*=", x_sol)
for i in range(1,len(xs)-1):
    ps_newton.append(np.log(abs(xs[i+1]-x_sol)) / np.log(abs(xs[i]-x_sol)))

x0=2.5
x1=x0-F(x0)/dF(x0)
errors = [abs(x1-x_sol)]
ps = []
for n in range(1,9):
    x2 = secant(x1,x0,F,dF)
    errors.append(abs(x2-x_sol))
    x0 = x1
    x1 = x2

for i in range(1,len(errors)-1):
    ps.append(np.log(errors[i+1]) / np.log(errors[i]))
print(ps)
plt.subplot(131)
#plt.semilogy()

plt.plot(range(2,9),ps,label="numerical convergence order secant")
plt.plot(range(2,steps+1),ps_newton,label="numerical convergence order newton")
plt.xlabel("steps")
plt.ylabel("error")
plt.grid()
plt.legend()
plt.subplot(132)
plt.semilogy()
plt.plot(range(9),errors,label="errors secant")
plt.plot(range(steps+1),abs(np.array(xs) - x_sol),label="errors newton")
plt.xlabel("steps")
plt.ylabel("error")
plt.grid()
plt.legend()
plt.subplot(133)
plt.semilogy()
plt.plot(np.array(range(9))*2,errors,label="errors secant")
plt.plot(np.array(range(steps+1))*3,abs(np.array(xs) - x_sol),label="errors newton")
plt.xlabel("function evaluations")
plt.ylabel("error")
plt.grid()
plt.legend()
plt.show()