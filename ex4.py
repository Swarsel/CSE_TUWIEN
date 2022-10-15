import math
import numpy as np
import matplotlib.pyplot as plt
# a)

def l(i, vals, x):
    return math.prod([(x - vals[j][0])/(vals[i][0] - vals[j][0]) if i != j else 1 for j in range(len(vals) )])

def p(x, vals):
    return sum([vals[i][1] * l(i, vals, x) for i in range(len(vals))])

def p0(x, f):
    vals = [(x[i], f[i]) for i in range(len(x))]
    p_0 = p(0, vals)
    n = len(vals)
    coef_np = np.polyfit([x[0] for x in vals], [x[1] for x in vals], n-1)
    for i in range(len(coef_np)):
        if coef_np[i] < 1.44 ** (-16):
            coef_np[i] = 0
    out = ""
    for i in range(n):
        if coef_np[i] == 0:
            continue
        out += f"{round(coef_np[i],2)} * x^{i} + "
    out = out[:-2]
    return p_0, out


# test a)
x1 = (0,0)
x2 = (1,2)
x3 = (4,8)
x = (0,1,4)
f = (0,2,8)
vals = [x1,x2,x3]
p_0, out = p0(x,f)
print(p_0)
print(out)

# b)
def h_i(i):
    return 2 ** (-i)

def N_init(f,m,n):
    N = np.empty((m+1, n+1))
    for j in range(m+1):
        for k in range(n+1):
            N[j][k] = np.polyval(np.polyfit(np.array(
                [h_i(o) for o in range(j, j+k+1)]), f(np.array([h_i(o) for o in range(j, j+k+1)])),k),0)
    return N

# c)

def f_h(h):
    return ((math.e ** h) -1)/(h)

m = 10
n = 2
N = N_init(f_h, m, n)
print(N)

plt.xlabel("h")
plt.ylabel("N[c] - 1")
plt.title("Convergence of N columns")
plt.loglog([h_i(o) for o in range(m + 1)], abs(N[: , 0] - 1), label = "column 1")
plt.loglog([h_i(o) for o in range(m + 1)], abs(N[: , 1] - 1), label = "column 2")
plt.loglog([h_i(o) for o in range(m + 1)], abs(N[: , 2] - 1), label = "column 3")

plt.legend()
plt.show()
