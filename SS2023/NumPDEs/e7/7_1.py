import numpy as np

def evalLegendre(x, k):
    if k == 1:
        return x
    if k == 0:
        return 1
    else:
        return ((2 * k - 1) * x * evalLegendre(x, k - 1) - (k - 1) * evalLegendre(x, k - 2)) / k


def eval_integrated_legendre(x, k):
    return (evalLegendre(x, k + 1) - evalLegendre(x, k - 1)) / (2 * k + 1)


def gauss_a(m, i, j, a=-1, b=1):
    x, w = np.polynomial.legendre.leggauss(m)
    # we are trying to solve a(phi1, phi2) = int (nabla phi1 nabla phi2)
    # basis functions are the integrated legendres -> use legendre here
    xshift = ((b - a) / 2) * x + ((b + a) / 2)
    return ((b - a) / 2) * sum(
        w * (evalLegendre(xshift, i) * evalLegendre(xshift, j)+eval_integrated_legendre(xshift, i)*eval_integrated_legendre(xshift, j)))


def gauss_f(m, i, a=-1, b=1):
    x, w = np.polynomial.legendre.leggauss(m)
    # we are trying to solve a(phi1, phi2) = int (nabla phi1 nabla phi2)
    # basis functions are the integrated legendres -> use legendre here
    xshift = ((b - a) / 2) * x + ((b + a) / 2)
    return ((b - a) / 2) * sum(w * (2 * np.sin(xshift) * evalLegendre(xshift, i)))

def exact(x):
    return np.sin(x)

def connectivity(N, p, i):
    CT = np.zeros((p+1,N + (p-1)*(N-1)))
    CT[0][i] = 1.0
    CT[1][i+1] = 1.0
    CT[2][i+4] = 1.0
    CT[3][i+7] = 1.0
    CT = CT.T
    #print(CT)
    return CT

# shorthands
L = evalLegendre
N = eval_integrated_legendre

p = 3  # order
N = np.linspace(0, np.pi, 4)
#print(N)
S = p + 1  # number of nodal functions
fT = []
AT = []
CT = []

for t in range(len(N)-1):
    A = np.zeros((S, S))
    for i in range(len(N)-1):
        for j in range(len(N)-1):
            A[i][j] = gauss_a(p, i + 2, j + 2, a=N[i], b=N[i+1])
    #print(A)
    AT.append(A)
    f = np.zeros((S,1))
    for i in range(len(N)-1):
        f[i][0] = gauss_f(p, i + 2, a=N[i], b=N[i+1])
    #print(f)
    fT.append(f)

    CT.append(connectivity(len(N), p, t))

A = np.zeros((len(N) + (p-1)*(len(N)-1), len(N) + (p-1)*(len(N)-1)))
for a, c in zip(AT, CT):
    #print(a)
    #print(c)
    #print(c @ a)
    #print(c @ a @ c.T)
    A += c @ a @ c.T

F = np.zeros((len(N) + (p-1)*(len(N)-1),1))
for f, c in zip(fT, CT):
    #print(f)
    #print(c)
    F += c @ f
print(A)
#print(F)

u = np.linalg.solve(A, F)
#print(u)
error = np.array([abs(ui-exact(n)) for ui,n in zip(u,N)])
normed = np.sqrt(error.T @ A @ error)
print(normed)
print(error)
