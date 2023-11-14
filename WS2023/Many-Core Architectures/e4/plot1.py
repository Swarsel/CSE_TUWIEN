import matplotlib.pyplot as plt
import numpy as np
# fix default matplotlib in nix trying to use qt when another library using qt is installed
from matplotlib import use as matplotlib_use
matplotlib_use("TkAgg")

N = [100, 300, 1000, 10000, 100000, 1000000, 3000000, 10000000, 100000000];

a=[1.44e-05, 1.83e-05, 2.35e-05, 2.34e-05, 0.0001262, 0.0010985, 0.0032829, 0.0107652, 0.104711]


b=[2.2e-05, 2.31e-05, 2.82e-05, 3.02e-05, 0.0001309, 0.0010647, 0.0031639, 0.0104985, 0.104741]


c=[1.07e-05, 1.32e-05, 1.34e-05, 2.17e-05, 0.0001244, 0.0011914, 0.0035546, 0.0118273, 0.118138]


d=[1.19e-05, 1.08e-05, 1.1e-05, 1.14e-05, 1.36e-05, 5.69e-05, 0.0001521, 0.0004789, 0.0046827]


plt.title("Comparison of summation kernels")
plt.xlabel("N")
plt.ylabel("Time [s]")
plt.semilogy(N, a,label="Shared Memory")
plt.semilogy(N, b,label="Warp Shuffles")
plt.semilogy(N, c,label="Warp Shuffles with 1 thread per entry")
plt.semilogy(N, d,label="Dot product")
plt.grid()
plt.legend()
plt.savefig("comp1.jpg")

plt.show()
