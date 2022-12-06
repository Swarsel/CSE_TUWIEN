import matplotlib.pyplot as plt
import numpy as np

k_max = 30
u_seq = [2]
error = [abs(np.pi - 2)]

for k in range(1, k_max + 1):
    # sqrt goes to zero, some decimal places are lost before
    u = 2 ** k * np.sqrt(2 * (1 - np.sqrt(1 - (2 ** (-k) * u_seq[k - 1]) ** 2)))
    print(f"k={k}, order or 2^k {np.floor(np.log10(2 ** k))} order of sqrt {np.floor(np.log10(np.sqrt(2 * (1 - np.sqrt(1 - (2 ** (-k) * u_seq[k - 1]) ** 2)))))}, error {abs(np.pi - u)}")
    print(f"k={k}, sqrt term {np.sqrt(2 * (1 - np.sqrt(1 - (2 ** (-k) * u_seq[k - 1]) ** 2)))}")

    u_seq.append(u)

    error.append(abs(np.pi - u))

print("minimum error at k = {}".format(error.index(min(error)) + 1))

plt.semilogy(range(k_max + 1), error)
plt.title("error of $\pi$ approx")
plt.xlabel("n")
plt.ylabel("error")
plt.show()