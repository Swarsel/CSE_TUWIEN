import numpy as np
import functions
import matplotlib.pyplot as plt

def composite_trapezoidal(a, b, N, f):
    h = (b - a) / N
    return sum([(h / 2) * (f(a + i * h) + f(a+ (i + 1) * h)) for i in range(N)])


def f1(x):
    return x ** 2


def f2(x):
    return abs(x)


def f3(x):
    if x < 1 / 3:
        return 1 / 2 * np.exp(x)
    else:
        return np.exp(x)


def f4(x):
    return np.sin(np.pi * x)


def f5(x):
    return np.sin(4 * np.pi * x)

errors_f1 = []
errors_f2 = []
errors_f3 = []
errors_f4 = []
errors_f5 = []
a,b = -1, 1
Ns = [2 ** i for i in range(1,21)]
for N in Ns:
    errors_f1.append(abs(2/3 - composite_trapezoidal(a,b, N, f1)))
    errors_f2.append(abs(1 - composite_trapezoidal(a,b, N, f2)))
    errors_f3.append(abs(np.e - 1/2 * (np.e ** (1/3) + np.e ** (-1) ) - composite_trapezoidal(a,b ,N, f3)))
    errors_f4.append(abs(0 - composite_trapezoidal(a,b, N, f4)))
    errors_f5.append(abs(0 - composite_trapezoidal(a,b, N, f5)))


plt.title("Compoisite Trapezoidal rule errors")
plt.xlabel("h")
plt.ylabel("error")
plt.loglog([(b - a) / N for N in Ns], errors_f1,  label="$f_1$")
plt.loglog([(b - a) / N for N in Ns], errors_f2,  label="$f_2$")
plt.loglog([(b - a) / N for N in Ns], errors_f3,  label="$f_3$")
plt.loglog([(b - a) / N for N in Ns], errors_f4,  label="$f_4$")
plt.loglog([(b - a) / N for N in Ns], errors_f5,  label="$f_5$")
plt.loglog([(b - a) / N for N in Ns], [1/N for N in Ns], linestyle="--", label="$h$")
plt.loglog([(b - a) / N for N in Ns], [1/N**2 for N in Ns],linestyle="--",  label="$h^2$")
plt.legend()
plt.show()
# the trapz error behaves like Ch²
# the box error behaves like Ch
