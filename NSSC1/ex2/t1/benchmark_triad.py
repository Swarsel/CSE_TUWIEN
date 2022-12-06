import matplotlib.pyplot as plt
import numpy as np

#i5-4300U
#Level 1 Cache	128 KB
#Level 2 Cache	512 KB
#Level 3 Cache	3 MB

with open("triad_out.txt", "r") as file:
    vals = np.array([[float(x) for x in line.split(", ")] for line in file])
n = vals[:,0]
flops_s = vals[:,-1]

cache = np.array([128, 512, 3072])
memory = (4*8)/1024
levels = np.array([level/memory for level in cache])

plt.title("Performance of vector triad for differing vector sizes")
plt.semilogx(n,flops_s)
plt.axvline(x=levels[0], color="green", label=f"L1 cache")
plt.axvline(x=levels[1], color="green", linestyle="--", label=f"L2 cache")
plt.axvline(x=levels[2], color="green", linestyle=":", label=f"L3 cache")
plt.xlabel("N")
plt.ylabel("Performance [FLOPS/s]")
plt.legend()
plt.show()
