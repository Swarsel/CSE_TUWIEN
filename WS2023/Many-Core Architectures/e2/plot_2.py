import matplotlib.pyplot as plt
import numpy as np

n=[100, 300, 1000, 10000, 100000, 1000000, 3000000]
a=[0.000016,0.000021,0.000021,0.000022,0.000024,0.000069,0.000171]
b=[0.000021, 0.000020, 0.000020, 0.000022, 0.000053, 0.000107, 0.000198]

plt.title("Comparison of dot products")
plt.xlabel("n")
plt.ylabel("Time [s]")
plt.loglog(n, a,label="2 kernels")
plt.loglog(n, b, label="1 kernel + CPU")
plt.legend()
plt.show()
