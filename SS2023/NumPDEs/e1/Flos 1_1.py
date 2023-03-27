import numpy as np

import matplotlib.pyplot as plt

N = 5
a = [[0 for i in range(N)]]
a[0][0]=1
b = [0]
for i in range(1,N-1):
    ins = []
    for j in range(N):
        if j == i-1 or j == i+1:
            ins.append(1)
        elif j == i:
            ins.append(-2)
        else:
            ins.append(0)
    a.append(ins)
    b.append(-1/(N-1)**2*(1+i/(N-1)))
a.append([0 for i in range(N)])
a[-1][-1]=1
b.append(0)
print(a)
print(b)
x = np.linalg.solve(a,b)
print(x)

points = np.linspace(0,1,100)
def u(x):
    return 2/3*x-1/2*x**2-1/6*x**3

plt.subplot(211)
plt.plot(points,u(points),label="exact")
discr = np.linspace(0,1,N)
errors = np.abs(x-u(discr))
plt.plot(discr,x,"--x",label="FD")
plt.legend()
plt.xlabel("x")
plt.ylabel("u(x)")

plt.subplot(212)
plt.plot(range(N),errors,"--x")
plt.xlabel("j")
plt.xticks(range(N))
plt.ylabel("error")
plt.show()