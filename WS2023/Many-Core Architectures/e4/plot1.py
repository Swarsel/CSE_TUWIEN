import matplotlib.pyplot as plt
import numpy as np

N = [100, 300, 1000, 10000, 100000, 1000000, 3000000];
a=[2.8e-06, 1.8e-06, 1.8e-06, 1.9e-06, 2.1e-06, 2.3e-06, 3.1e-06, 5.9e-06, 9.8e-06]


b=[3.1e-06, 1.8e-06, 2e-06, 2.1e-06, 2.1e-06, 2.4e-06, 3.2e-06, 5.6e-06, 1.04e-05]


c=[2.5e-06, 1.7e-06, 1.7e-06, 1.8e-06, 2.3e-06, 2.4e-06, 3.2e-06, 5.5e-06, 9.4e-06]


d=[2.8e-06, 1.8e-06, 1.6e-06, 1.6e-06, 2.3e-06, 2.7e-06, 4.7e-06, 5.8e-06, 1.04e-05]





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
