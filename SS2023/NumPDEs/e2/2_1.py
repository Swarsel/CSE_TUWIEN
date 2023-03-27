import numpy as np

import matplotlib.pyplot as plt

for N in [4,8,16,32]:
    a = np.zeros((N+1,N+1))
    a[0][0] = 1
    a[-1][-1] = 1
    b1, b2 = [0], [0]
    for i in range(1,N):
        x_i = i/(N)
        for j in range(N+1):
            if j == i-1 or j == i+1:
                a[i][j] = 1
            elif j == i:
                a[i][j] = -2
        b1.append(-1/(N)**2 * (1+x_i))
        b2.append(-1/(N)**2 * (2 * x_i**2 + 3 * x_i - 4/3))
    b1.append(0)
    b2.append(0)

    print(f"a: {a}")
    print(f"b1: {b1}")
    print(f"b2: {b2}")
                  
    x1 = np.linalg.solve(a,b1)
    x2 = np.linalg.solve(a,b2)
    print(f"ui 1: {x1}")
    print(f"ui 2: {x2}")

    points = np.linspace(0,1,100)
    def u1(x):
        return 2/3 * x - 1/2 * x**2- 1/6 * x**3

    def u2(x):
        return 2/3 * x**2 - 1/2 * x**3 - 1/6 * x**4

    plt.subplot(2, 2, 1)
    plt.plot(points,u1(points),label="exact")
    discr = np.linspace(0,1,N+1)
    errors = np.abs(x1-u1(discr))
    plt.plot(discr,x1,"--x",label="FD")
    plt.legend()
    plt.xlabel("x")
    plt.ylabel("$u_1(x)$")

    plt.subplot(2, 2, 2)
    plt.plot(range(N+1),errors,"--x")
    plt.xlabel("j")
    plt.xticks(range(N+1))
    plt.ylabel("error")

    plt.subplot(2, 2, 3)
    plt.plot(points,u2(points),label="exact")
    discr = np.linspace(0,1,N+1)
    errors = np.abs(x2-u2(discr))
    plt.plot(discr,x2,"--x",label="FD")
    plt.legend()
    plt.xlabel("x")
    plt.ylabel("$u_2(x)$")

    plt.subplot(2, 2, 4)
    plt.plot(range(N+1),errors,"--x")
    plt.xlabel("j")
    plt.xticks(range(N+1))
    plt.ylabel("error")
    plt.show()
