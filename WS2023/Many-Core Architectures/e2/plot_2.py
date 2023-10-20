import matplotlib.pyplot as plt
import numpy as np

n=[100, 300, 1000, 10000, 100000, 1000000, 3000000]
a=[0.000007, 0.000006, 0.000007, 0.000010, 0.000049, 0.000533, 0.003983]
b=[0.000014, 0.000013, 0.000016, 0.000044, 0.000332, 0.003330, 0.013234]

plt.title("Comparison of dot products")
plt.xlabel("n")
plt.ylabel("Time [s]")
plt.semilogx(n, a,label="2 kernels")
plt.semilogx(n, b, label="1 kernel + CPU")
plt.legend()
plt.show()
