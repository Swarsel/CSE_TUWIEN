import matplotlib.pyplot as plt
import numpy as np

k_max = 30
u_seq = [2]
abs_err = []

for k in range(1, k_max + 1):
    # sqrt goes to zero, some decimal places are lost before
    u = 2 ** k * np.sqrt(2 * (1 - np.sqrt(1 - (2 ** (-k) * u_seq[k - 1]) ** 2)))
    print(f"k={k}, order or 2^k {np.floor(np.log(2 ** k))} order of sqrt {np.floor(np.log(np.sqrt(2 * (1 - np.sqrt(1 - (2 ** (-k) * u_seq[k - 1]) ** 2)))))}, error {abs(np.pi - u)}")

    u_seq.append(u)

    abs_err.append(abs(np.pi - u))

print("minimum error at k = {}".format(abs_err.index(min(abs_err)) + 1))

plt.semilogy(range(1, k_max + 1), abs_err)
plt.title("error of pi approx")
plt.xlabel("number of elements of the sequence")
plt.ylabel("absolute error")
plt.show()