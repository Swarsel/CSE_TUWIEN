import matplotlib.pyplot as plt
from math import pi, sin

xs = [i for i in range(-100,100)]
for i in range(len(xs)):
    xs[i] = xs[i] / 100

print(xs)
sin1 = [pi* i - (1/6) * pi ** 3 * i ** 3 for i in xs]
sin2 = [2/pi + (10 * (pi ** 2 -12) * (6 * i ** 2 - 6 * i + 1))/(pi ** 3) for i in xs]

plt.xlabel("x")
plt.ylabel("f(x)")
plt.title("plot of approxes of sin")
plt.plot(xs, sin1, label = "Taylor")
plt.plot(xs, sin2, label = "bestapprox")
plt.plot(xs, [sin(pi * x) for x in xs], label = "sin")

plt.legend()
plt.show()
