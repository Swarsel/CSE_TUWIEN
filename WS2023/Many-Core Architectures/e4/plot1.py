import matplotlib.pyplot as plt
import numpy as np

N = [100, 300, 1000, 10000, 100000, 1000000, 3000000];

a=[2.9e-06, 1.8e-06, 1.9e-06, 2.1e-06, 2.2e-06, 2.5e-06, 3.1e-06, 5.9e-06, 6.8e-06]


b=[2.9e-06, 2e-06, 1.8e-06, 2.1e-06, 1.9e-06, 2.2e-06, 3e-06, 5.7e-06, 1e-05]


c=[2.5e-06, 1.8e-06, 1.7e-06, 2e-06, 2e-06, 2.2e-06, 3.3e-06, 5.2e-06, 6.3e-06]


d=[2.9e-06, 1.8e-06, 1.8e-06, 1.8e-06, 2.2e-06, 2.7e-06, 5.1e-06, 5.6e-06, 1.1e-05]

plt.title("Comparison of summation kernels")
plt.xlabel("N")
plt.ylabel("Time [s]")
plt.loglog(N, a,label="Shared Memory")
plt.loglog(N, b,label="Warp Shuffles")
plt.loglog(N, c,label="Warp Shuffles with 1 thread per entry")
plt.loglog(N, d,label="Dot product")
plt.grid()
plt.legend()
plt.savefig("comp1.jpg")

plt.show()
