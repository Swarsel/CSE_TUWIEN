import numpy as np
from time import time

def S(N):
    return sum([1/(n ** 2) for n in range(1,N+1)])

def approx_S(N):
    return [1, 1/N, 1/(N ** 2)]

a = np.array([approx_S(N) for N in [100, 1000, 10000]])
b = np.array([S(N)for N in [100, 1000, 10000]])

x = np.linalg.solve(a,b)
limit = (np.pi ** 2) / 6
abserror = abs(x[0] - limit)
relerror = abserror/limit*100

print(f"Absolute error: {abserror}\n"
      f"Relative error: {relerror}%")