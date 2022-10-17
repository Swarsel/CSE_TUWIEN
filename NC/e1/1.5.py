import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(0,1.6,100)

plt.plot(x, [np.sin(v) for v in x], label="$\sin(x)$")
plt.plot(x, [v for v in x], label="$a=0$")
plt.plot(x, [1 - 1/2 * (v - (np.pi)/2) ** 2 for v in x], label="$a= \pi/2$")

plt.legend()
plt.show()