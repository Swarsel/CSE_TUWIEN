from math import e
import matplotlib.pyplot as plt

def T(N,f):
    h = 1/N
    return sum([(h / 2) * (f(i*h) + f((i+1)*h)) for i in range(N)])

def E(N,C):
    h = 1/N
    return C * (h ** 2)

def C(N, f1, f2):
    h = 1/N
    #T(h) - T(h/2) = Ch^2 - Ch^2/4 = (3Ch^2)/4
    #C = 4/3 / h^2 * (T(h) - T(h/2))
    #C = 4/3 * N^2 * (T(h) - T(h/2))
    return (4/ (3 * (h ** 2))) * (f1 - f2)

def f_e(x):
    return e ** x

def N_i(i):
    return 2 ** i

r_trapz = []
h_i = []
for it in range(1,11):
    h_i.append( 1 / N_i(it))
    C_i = C(N_i(it), T(2*N_i(it),f_e), T(N_i(it),f_e))
    E_i = E(N_i(it),C_i)
    print(f"i:{it}; C:{C_i}, E:{E_i}")
    r_trapz.append((e - 1 - T(N_i(it), f_e))/E_i)
    #r_trapz.append(abs((e - 1 - T(N_i(it), f_e))/(E_i*C_i)))

plt.title("Trapezoidal rule ratio of error to estimator")
plt.xlabel("h")
plt.ylabel("ratio r")
plt.semilogx(h_i, r_trapz)
#shows the difference in C
plt.show()
