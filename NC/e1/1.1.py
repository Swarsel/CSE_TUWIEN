from math import e
import matplotlib.pyplot as plt

def T(N,f):
    h = 1/N
    return sum([(h / 2) * (f(i*h) + f((i+1)*h)) for i in range(N)])

def R(N,f):
    h = 1/N
    return sum([h * f(i*h) for i in range(N)])

def f_e(x):
    return e ** x

def N_i(i):
    return 2 ** i

errors_trapz = []
errors_box = []
for it in range(1,11):
    errors_trapz.append(abs(e - 1 - T(N_i(it), f_e)))
    errors_box.append(abs(e - 1 - R(N_i(it), f_e)))

plt.title("Box rule and trapezoidal rule errors")
plt.xlabel("N")
plt.ylabel("Absolute error")
plt.loglog([1/N_i(i) for i in range(1,11)], errors_trapz,  label="trapz")
plt.loglog([1/N_i(i) for i in range(1,11)], errors_box,  label="box")
plt.loglog([1/N_i(i) for i in range(1,11)], [1/N_i(i) for i in range(1,11)],linestyle="--", label="h")
plt.loglog([1/N_i(i) for i in range(1,11)], [1/(N_i(i)**2) for i in range(1,11)],linestyle="--", label="h²")
plt.legend()
plt.show()
# the trapz error behaves like Ch²
# the box error behaves like Ch

