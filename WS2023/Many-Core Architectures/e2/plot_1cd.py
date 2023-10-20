import matplotlib.pyplot as plt
import numpy as np

n=[100,300,1000,10000,100000,1000000,3000000]
t=[0.000009,0.000008,0.000008,0.000009,0.000011,0.000077,0.000229]

plt.title("Execution times, vector addition")
plt.xlabel("n")
plt.ylabel("Time [s]")
plt.loglog(n, t)
plt.show()
