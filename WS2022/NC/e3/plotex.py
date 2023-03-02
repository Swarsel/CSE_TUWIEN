import matplotlib.pyplot as plt
import numpy as np

x1 = np.linspace(-1, 1/3, 100)
x2 = np.linspace(1/3, 1, 100)

plt.plot(x1, [1/2 * np.e ** (x) for x in x1])
plt.plot(x2, [np.e ** (x) for x in x2])
plt.show()