import matplotlib.pyplot as plt
import math

k_max = 30
u_seq = [2]
abs_err = []

for k in range(1, k_max + 1):
    u = 2 ** k * math.sqrt(2 * (1 - math.sqrt(1 - (2 ** (-k) * u_seq[k - 1]) ** 2)))
    u_seq.append(u)

    abs_err.append(abs(math.pi - u))

print("minimum error at k = {}".format(abs_err.index(min(abs_err)) + 1))

plt.semilogy(range(1, k_max + 1), abs_err)
plt.title("error of pi approx")
plt.xlabel("number of elements of the sequence")
plt.ylabel("absolute error")
plt.show()