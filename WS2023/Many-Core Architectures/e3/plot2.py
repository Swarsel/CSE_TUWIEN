import matplotlib.pyplot as plt
import numpy as np

n=[64, 128, 256, 512, 1024, 2048, 4096]
b=[2.84939, 7.71012, 9.70904, 17.3318, 32.768, 55.7383, 71.621]
c=[3.44926, 9.70904, 15.8875, 33.2881, 68.2001, 93.8586, 125.379]
d=[4.36907, 20.1649, 80.6597, 174.763, 275.036, 305.04, 304.694]

plt.title("Comparison of matrix transposition kernels")
plt.xlabel("N")
plt.ylabel("Effective Bandwidth [GB/s]")
plt.semilogy(n, d,label="Unoptimized in-place ")
plt.semilogy(n, c,label="Two matrices")
plt.semilogy(n, b,label="Optimized in-place")
plt.grid()
plt.legend()
plt.savefig("comp2.jpg")

plt.show()
