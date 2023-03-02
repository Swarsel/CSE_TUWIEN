import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as opt

def interpol_poly(coefs_passed, x):
    # calculates function value of an arbitrary interpolating polynomial function
    out = 0
    xpot = 1
    coefs = np.flipud(coefs_passed)
    for coef in coefs:
        out += coef * xpot
        xpot *= x
    return out

# values given by task
vals_x = [0, 1, 2]
vals_y = [1, np.e, np.e ** 2]

# compute coefficients for interpolating polynomial of order two
quadratic_poly_coef = np.polyfit(vals_x, vals_y, 2)

# make grid and compute values of interpolating polynomial
xs = np.linspace(0,2,200)
quadratic_polys = [interpol_poly(quadratic_poly_coef, x) for x in xs]
# compute the lower function at x=1.2 and the error there to plot it later
lower_f_1_2 = min(np.e ** (1.2), interpol_poly(quadratic_poly_coef, 1.2))
error_1_2 = abs(np.e ** (1.2) - interpol_poly(quadratic_poly_coef, 1.2))
print(error_1_2)
# error at x=1.2 : 0.09612036445581262
plt.title("Quadratic interpolation of $e^x$")
plt.xlabel("$x$")
plt.ylabel("$f(x)$")
plt.plot(xs, [y for y in quadratic_polys], label= "quadratic polynomial")
plt.plot(xs, [np.e ** x for x in xs], label = "$e^x$", linestyle = "--")
# plot the error
plt.plot([1.2, 1.2], [lower_f_1_2 , lower_f_1_2 + error_1_2], label=f"Error at $x=1.2$: {error_1_2}")
plt.legend()
plt.show()


### pointwise error
"""
f(x) = e^x, x0 = 0, x1= 1, x2= 2
n = 2
so:
|f(x) - p(x)| leq |omega_3 (x) f'''(xi)/3!| leq (e^x)/6 |(x-0)(x-1)(x-2)|
omega_3 = (x-0)(x-1)(x-2)
f'''(xi) = e^xi
on [0,2] |f'''(x)| = |e^x| leq |e^2|
"""

# plug in numbers
pointwise_errors = [(abs(np.e ** 2)/6) * abs(x * (x-1) * (x-2)) for x in xs]

# plot
plt.title("pointwise upper bound of the error of $e^x - p(x)$ on $[0,2]$")
plt.xlabel("$x$")
plt.ylabel("$f(x)$")
plt.plot(xs, [abs(np.e ** x - interpol_poly(quadratic_poly_coef, x)) for x in xs], label = "$e(x)$", linestyle = "--")
plt.plot(xs, pointwise_errors, label = "$e_{\mathrm{upper}}(x)$")
plt.legend()
plt.show()

### maximal error
"""
||f-p||_infty[a,b] (b-a)^(n+1) (||f^(n+1)||_infty[a,b])((n+1)!)
a = 0
b = 2
n = 2
||f^(n+1)||_infty[a,b] = e^3
maximum_error = (2 ** 3) * (np.e ** 2)/(6)
"""
# plug in numbers
a, b = 0, 2
n = 2
maximum_error = (b - a) ** (n+1) * (np.e ** 2)/(np.math.factorial(n+1))
# plot
plt.title("Upper bound for the error of  $e^x - p(x)$ on $[0,2]$")
plt.xlabel("$x$")
plt.ylabel("$f(x)$")
plt.plot(xs, [abs(np.e ** x - interpol_poly(quadratic_poly_coef, x)) for x in xs], label = "$e(x)$", linestyle = "--")
plt.plot(xs, [maximum_error for _ in xs], label = "$e_{\mathrm{max}}(x)$")
plt.legend()
plt.show()