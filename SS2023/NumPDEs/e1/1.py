import numpy as np

import matplotlib.pyplot as plt

N = 4
a = np.zeros((N+1,N+1))
a[0][0] = 1
a[-1][-1] = 1
b = [0]
for i in range(1,N):
    for j in range(N+1):
        if j == i-1 or j == i+1:
            a[i][j] = 1
        elif j == i:
            a[i][j] = -2
    b.append(-1/(N)**2*(1+i/(N)))
b.append(0)

print(f"a: {a}")
print(f"b: {b}")
x = np.linalg.solve(a,b)
print(f"ui: {x}")

points = np.linspace(0,1,100)
def u(x):
    return 2/3*x-1/2*x**2-1/6*x**3

plt.subplot(211)
plt.plot(points,u(points),label="exact")
discr = np.linspace(0,1,N+1)
errors = np.abs(x-u(discr))
plt.plot(discr,x,"--x",label="FD")
plt.legend()
plt.xlabel("x")
plt.ylabel("u(x)")

plt.subplot(212)
plt.plot(range(N+1),errors,"--x")
plt.xlabel("j")
plt.xticks(range(N+1))
plt.ylabel("error")
plt.show()
