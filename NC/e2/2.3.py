import numpy as np
import matplotlib.pyplot as plt

### a)

def neville(x,f,x_eval):
    if len(x) != len(f):
        print(f"Warning: {x} and {f} do not have the same dimensions!")
    n = len(x)
    p = np.zeros((n, n))

    # populate first column
    p[:, 0] = f
    # iterate first over columns, so that we only use defined values
    for m in range(1,n):
        # columns get smaller by 1 each iteration
        for j in range(0,n-m):
            p[j][m] = ((x_eval - x[j]) * p[j+1][m-1] - (x_eval - x[j+m]) * p[j][m-1])/(x[j+m] - x[j])

    return p
"""
# compare with example from lecture slides
x_test = [0,1,3]
y_test = [1,3,2]
x_eval_test = 2
print(f"Neville scheme for x={x_test}, f(x)={y_test} at evaluation point x={x_eval_test}: \n"
      f"{neville(x_test,y_test,x_eval_test)}")
print()


def D_lecture(f, hs, x_eval):
    return [(f(x_eval + h) - f(x_eval))/(h) for h in hs]

def f_ex(x):
    return np.e ** x

x_eval_test = 0
hs = [2 ** (-i) for i in range(9)]
D_ex = D_lecture(f_ex, hs, x_eval_test)
neville_test = neville(hs, D_ex, x_eval_test)
print(f"Test neville (slides):\n"
      f"{neville_test}")
print()
error_test = f_ex(x_eval_test) - neville_test
print(f"Test neville error:\n"
      f"{error_test}")
print()
"""

def D(f, h, x_eval):
    return (f(x_eval + h) - f(x_eval - h))/(2 * h)

def f_tan(x):
    return np.tan(x)

x_eval = np.pi / 4

hs = [2 ** (-i) for i in range(11)]
D_tan = [D(f_tan, h, x_eval) for h in hs]

neville_D = neville(hs, D_tan, 0)
print(f"Neville scheme of symmetric difference quotient, f(x) = tan(x) and x_0= pi/4:\n"
      f"{neville_D}")
print()

def f_dtandx(x):
    return 1/(np.cos(x) ** 2)

neville_D_errors = abs(neville_D - f_dtandx(x_eval))
"""print(f"Errors of neville:\n"
      f"{neville_D_errors}")
print()"""

neville_column0 = [neville_D_errors[i][0] for i in range(len(hs))]
neville_column1 = [neville_D_errors[i][1] for i in range(len(hs)-1)]
neville_column2 = [neville_D_errors[i][2] for i in range(len(hs)-2)]
neville_column3 = [neville_D_errors[i][3] for i in range(len(hs)-3)]
h2 = [h ** 2 for h in hs]
h3 = [h ** 3 for h in hs]
h4 = [h ** 4 for h in hs]

plt.title("Neville scheme error convergence")
plt.xlabel("$h$")
plt.ylabel("error")
plt.loglog(hs, neville_column0, color = "red", label= "1st column")
plt.loglog(hs[:-1], neville_column1, color = "cyan", label= "2nd column")
plt.loglog(hs[:-2], neville_column2, color = "blue", label= "3rd column")
plt.loglog(hs[:-3], neville_column3, color = "green", label= "4th column")
plt.loglog(hs, h2, color = "cyan", label= "$h^2$", linestyle = "--")
plt.loglog(hs, h3, color = "blue", label= "$h^3$", linestyle = "--")
plt.loglog(hs, h4, color = "green", label= "$h^4$", linestyle = "--")
plt.loglog(hs, [f_tan(x) for x in hs], color = "black", label= "$f(x) = \\tan(x)$", linestyle = "--")
plt.legend()
plt.show()