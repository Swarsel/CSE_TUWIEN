import matplotlib.pyplot as plt
import numpy as np

xs = np.linspace(-6, 6, 100)


def curv(x):
    return (-np.sin(x)) / ((1 + np.cos(x) ** 2) ** (3 / 2))

ys = [curv(x) for x in xs]

plt.plot(xs, ys)
plt.show()