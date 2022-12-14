import numpy as np
import matplotlib.pyplot as plt

def interpol_poly(coefs_passed, x):
    # calculates function value of an arbitrary interpolating polynomial function
    out = 0
    xpot = 1
    coefs = np.flipud(coefs_passed)
    for coef in coefs:
        out += coef * xpot
        xpot *= x
    return out

def piecewise(N, lower, higher, f, n):
    # calculates function value of piecewise polynomial of degree n
    xs = []
    ys = []
    h = (higher - lower) / N
    # partition in N subintervals
    subintervals = [(lower + j * h, lower + (j + 1) * h) for j in range(N)]
    for interval in subintervals:
        # x_(i,j) = t_j + ih/n, i = 0,...,n
        vals_x = [interval[0] + 1/n * i * h for i in range(n+1)]
        # append all but last x value to output xs (otherwise we would have them twice)
        xs += vals_x[:-1]
        # calculate function values at these points
        vals_y = [f(x) for x in vals_x]
        # calculate coefficients of interpolating polynomial
        coefs = np.polyfit(vals_x, vals_y, n)
        # again append all but last function value
        ys += [interpol_poly(coefs, x) for x in vals_x[:-1]]
    # we now have all values needed except the very last ones, add them manually
    xs.append(vals_x[-1])
    ys.append(interpol_poly(coefs, vals_x[-1]))

    return (xs, ys)

def e_x(x):
    return np.e ** x

# values for comparison e^x
xs_base = np.linspace(0,2,100)
ys_base = [e_x(x) for x in xs_base]

for N in [2,5,100]:
    xs, ys = piecewise(N, 0, 2, e_x, 1)
    plt.title(f"Piecewise function as approximation of $e^x$, $n=1, N={N}$")
    plt.xlabel("$x$")
    plt.ylabel("$f(x)$")
    plt.plot(xs,ys, label="piecewise function")
    plt.plot(xs_base, ys_base, label="$e^x$", linestyle = "--")
    plt.legend()
    plt.show()