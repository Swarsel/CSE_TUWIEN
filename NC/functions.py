import numpy as np
import math


def neville(x, f, x_eval):
    if len(x) != len(f):
        print(f"Warning: {x} and {f} do not have the same dimensions!")
    n = len(x)
    p = np.zeros((n, n))

    # populate first column
    p[:, 0] = f
    # iterate first over columns, so that we only use defined values
    for m in range(1, n):
        # columns get smaller by 1 each iteration
        for j in range(0, n - m):
            p[j][m] = ((x_eval - x[j]) * p[j + 1][m - 1] - (x_eval - x[j + m]) * p[j][m - 1]) / (x[j + m] - x[j])

    return p


def lagrange_l(i, knots, x):
    # returns ith lagrange polynomial at a point x
    return math.prod([(x - knots[j]) / (knots[i] - knots[j]) if i != j else 1 for j in range(len(knots))])


def lagrange(x, knots):
    # returns lagrance interpolating polynomial at point x
    return sum([knots[i] * lagrange_l(i, knots, x) for i in range(len(knots))])


def least_squares_poly(vals_x, vals_y, order, x):
    # calculates function value of least squares interpolating polynomial function
    quadratic_poly_coef = np.polyfit(vals_x, vals_y, order)
    out = 0
    xpot = 1
    coefs = np.flipud(quadratic_poly_coef)
    for coef in coefs:
        out += coef * xpot
        xpot *= x
    return out


def lebesgue_const(x_domain, x_points):
    # returns lebesgue constant over a domain (e.g. [-5,5]) for a distribution of points
    elements = []
    for x in x_domain:
        elements.append(sum([abs(lagrange_l(i, x_points, x)) for i in range(len(x_points))]))

    return max(elements)
