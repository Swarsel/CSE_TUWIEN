import numpy as np
from time import time

def S(N):
    return sum([1/n for n in range(1,N+1)])

#alternative implementation since the first one encounters a memoryerror for high N
def S_alt(N):
    sum = 0
    for n in range(1,N+1):
        sum += 1/n
    return sum

def S_approx_row_vector(N):
    return [1, 1/N, 1/(N ** 2)]

def S_approx(coefs, N):
    return np.log(N) + coefs[0] + coefs[1]/N + coefs[2]/(N ** 2)

# bring log(N) term to this side first
a = np.array([S_approx_row_vector(N) for N in [10, 100, 1000]])
b = np.array([S(N) - np.log(N) for N in [10, 100, 1000]])

x = np.linalg.solve(a,b)

for N in [10 ** 6, 10 ** 7]:
    print(f"Error for {N}: \nabsolute: {abs(S(N) - S_approx(x, N))}, \nrelative: {abs(S(N) - S_approx(x, N))/S(N)} - true value: {S(N)}")
    print()

for N in [10 ** 8, 10 ** 9]:
    t0 = time()
    S_approx(x, N)
    t0 = time() - t0
    print(f"Time of S_approx(N) for N={N}: {t0}")

try:
    for N in [10 ** 8, 10 ** 9]:
        t0 = time()
        S_alt(N)
        t0 = time() - t0
        print(f"Time of S(N) for N={N}: {t0}")
except MemoryError:
    print(f"MemoryError encountered for S(N) with N=10^{np.log10(N)}")
