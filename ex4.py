import math
import numpy

def l(i, vals, x):
    return math.prod([(x - vals[j][0])/(vals[i][0] - vals[j][0]) if i != j else 1 for j in range(len(vals) )])

def p(x, vals):
    return sum([vals[i][1] * l(i, vals, x) for i in range(len(vals))])

def p0(x, f):
    vals = [(x[i], f[i]) for i in range(len(x))]
    p_0 = p(0, vals)
    n = len(vals)
    coef_numpy = numpy.polyfit([x[0] for x in vals], [x[1] for x in vals], n-1)
    for i in range(len(coef_numpy)):
        if coef_numpy[i] < 1.44 ** (-16):
            coef_numpy[i] = 0
    out = ""
    for i in range(n):
        if coef_numpy[i] == 0:
            continue
        out += f"{round(coef_numpy[i],2)} * x^{i} + "
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

def h_i(i):
    return 2 ** (-i)

def N_init(f,m,n):
    N = numpy.empty((m+1, n+1))
    for m in range(m):
        for n in range(n):
            N[m][n] = p_0(f)[0]


def f_h(h):
    return ((math.e ** h) -1)/(h)

b = N_init(1, 2, 2)
print(b)
