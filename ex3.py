import math
import numpy

def l(i, vals, x):
    return math.prod([(x - vals[j][0])/(vals[i][0] - vals[j][0]) if i != j else 1 for j in range(len(vals) )])

def p(x, vals):
    return sum([vals[i][1] * l(i, vals, x) for i in range(len(vals))])


# comparison
x1 = (0,0)
x2 = (1,2)
x3 = (4,8)

vals = [x1,x2,x3]

pc = p(2, vals)
print(pc)
coef_numpy = numpy.polyfit([x[0] for x in vals], [x[1] for x in vals], 2)
print(coef_numpy)
print(numpy.polyval(coef_numpy, 2))