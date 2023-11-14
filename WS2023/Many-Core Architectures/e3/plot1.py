import matplotlib.pyplot as plt
import numpy as np
# fix default matplotlib in nix trying to use qt when another library using qt is installed
from matplotlib import use as matplotlib_use
matplotlib_use("TkAgg")

n1=list(range(64))
n2=list(range(1,64))

stride = [331.249, 150.186, 99.9588, 77.0723, 74.0364, 72.8783, 71.0894, 71.7206, 67.8541, 65.8418, 63.508, 64.3956, 58.7835, 56.219, 53.9029, 53.5523, 47.923, 46.0946, 44.6251, 45.6308, 43.5076, 42.4412, 41.1823, 41.7728, 39.8886, 39.3653, 38.8042, 38.8005, 37.4169, 37.3396, 37.4821, 40.863, 37.1987, 36.0475, 35.2281, 34.9333, 34.215, 33.965, 33.74, 34.1802, 33.2009, 32.7767, 32.3897, 32.1461, 31.5227, 31.2194, 30.8916, 30.8852, 30.8494, 31.1123, 31.26, 31.9736, 31.9052, 32.2084, 32.5014, 33.2923, 33.1276, 33.724, 34.2667, 34.5692, 35.4389, 35.7397, 36.7857]
skip=[331.391, 302.809, 303.122, 303.582, 308.376, 291.135, 291.407, 290.913, 312.622, 292.947, 293.406, 293.395, 314.05, 294.265, 294.255, 294.071, 326.615, 294.302, 293.894, 294.363, 314.144, 293.219, 293.294, 293.459, 312.512, 290.98, 291.319, 290.726, 307.408, 303.321, 304.044, 303.971, 331.482, 302.476, 302.37, 302.16, 307.803, 290.367, 290.511, 290.346, 311.514, 292.59, 292.579, 292.383, 313.365, 293.266, 293.373, 293.105, 326.193, 293.337, 293.083, 293.126, 313.279, 292.387, 292.672, 292.519, 311.3, 290.269, 290.469, 290.427, 307.886, 302.332, 302.797, 302.309]



plt.title("Comparison of stride and skip implementations")
plt.xlabel("k")
plt.ylabel("Effective Bandwidth [GB/s]")
plt.semilogy(n2, stride,label="Stride")
plt.semilogy(n1, skip,label="Skip")
plt.grid()
plt.legend()
plt.savefig("comp1.jpg")

plt.show()
