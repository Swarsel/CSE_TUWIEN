import matplotlib.pyplot as plt
import numpy as np

def f(x):
    return -(x**2)/2 - (x**3)/6 + (2*x)/3

N = 4
xgo = np.linspace(0,1,100)
ygo = [f(x) for x in xgo]
xfin = np.linspace(0,1,N+1)
ygo2 = [f(x) for x in xfin]
yfin = [0, 17/128, 24/128, 19/128, 0]
plt.plot(xgo, ygo, label="exact")
plt.plot(xfin, yfin, label="finite difference approx.")
plt.legend()
plt.show()

plt.plot(xfin, [abs(yfin[i] - ygo2[i]) for i in range(5)])
plt.xlabel("x")
plt.ylabel("error")
plt.show()
