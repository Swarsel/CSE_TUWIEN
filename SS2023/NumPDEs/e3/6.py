import numpy as np

import matplotlib.pyplot as plt

def calc_u(N,h):
    A = np.zeros((N-1,N-1))
    b = []

    for i in range(N-1):
        if i < N - 2:
            A[i, i+1] = -1
        A[i, i] = 2
        if i > 0:
            A[i, i-1] = -1
        b.append(3*(i+1)*h**3)
    b[-1] += 1


    #print(f"A: {A}")
    #print(f"b: {b}")
    u = np.linalg.solve(A,b)
    #print(f"ui: {u}")
    return u


def sol(x):
    return (-1/2)*(x**3) + (3/2)*x

def error(N,h):
    u=calc_u(N,h)
    error = [abs(sol((i*h+(i+1)*h)/2)-((u[i]+u[i+1])/2)) for i in range(N-2)]
    return max(error)

N = 10
h = 1/N

u = calc_u(N, h)
points = np.zeros(N-1)
points = [i*h for i in range(1,N)]

plt.subplot(211)
plt.plot(points, [sol(x) for x in points], linestyle=":", label="exact")
plt.plot(points, u, "x", label="approx")
plt.legend()
plt.xlabel("x")
plt.ylabel("u(x)")

#steps = [i for i in range(3, 50)]
h_i = [1/l for l in steps]
error = [error(n, h) for n,h in zip(steps,h_i)]
#print(error)

plt.subplot(212)
plt.plot(steps, error)
plt.xlabel("n")
plt.ylabel("error")
plt.show()
