import numpy as np
import matplotlib.pyplot as plt

np.set_printoptions(precision=3)


# legendre polynomial transformed to (0, 1)
def L(x, k):
    if k == 1:
        return 2 * x - 1
    elif k == 0:
        return np.ones_like(x)
    else:
        return (2 - 1 / k) * (2 * x - 1) * L(x, k - 1) - (1 - 1 / k) * L(x, k - 2)


# integrated legendre polynomial transformed to (0, 1)
def N(x, i):
    if i == 0:
        return 2 * x - 1
    else:
        return (L(x, i + 1) - L(x, i - 1)) / (2 * i + 1)


class FE:
    def __init__(self, nodes, F, p, sys_dim):
        self.nodes = nodes
        # Affine linear map
        self.F = F
        # Size of finite element
        self.h = F[1]
        # Number of Gauss points
        self.m = p + 1
        # maximal polynomial degree of basis
        self.p = p
        # dimension of basis
        self.dim = p + 1
        # total size of FEM System
        self.N = sys_dim

        # Defining the basis functions and their derivatives
        self.basis = [lambda x: 1 - x, lambda x: x]
        self.dbasis = [lambda x: -np.ones_like(x), lambda x: np.ones_like(x)]
        for j in range(p - 1):
            self.basis += [lambda x, j=j: N(x, j + 1)]
            self.dbasis += [lambda x, j=j: 2 * L(x, j + 1)]

    # Computes one entry of the element stiffness matrix
    def stiffA(self, i, j):
        x, w = np.polynomial.legendre.leggauss(self.m)
        # Transformation of Gauss points and weights to (0,1)
        x = (x + 1) / 2
        w /= 2

        SUM1 = np.sum(self.dbasis[i](x) * self.dbasis[j](x) * w) / self.h
        SUM2 = self.h * np.sum((self.basis[i](x) * self.basis[j](x)) * w)
        return (SUM1 + SUM2)

    # Computes one entry of the element load vector
    def loadF(self, i):
        x, w = np.polynomial.legendre.leggauss(self.m + 6)
        # Transformation of Gauss points and weights to (0,1)
        x = (x + 1) / 2
        w /= 2

        return (
                self.h
                * np.sum(2 * np.sin(self.F[0] + self.F[1] * x) * self.basis[i](x) * w)
        )

    # Returns the connectivity matrix for this FE
    @property
    def ConnectivityMatrix(self):
        C = np.zeros((self.N, self.dim))
        for i, n in enumerate(self.nodes):
            C[n, i] = 1
        return C

    # Returns the stiffness matrix for this FE
    @property
    def StiffnessMatrix(self):
        A = np.zeros((self.dim, self.dim))
        for i in range(self.dim):
            for j in range(self.dim):
                A[i, j] = self.stiffA(i, j)

        return A

    # Returns the load vector for this FE
    @property
    def LoadVector(self):
        F = np.zeros(self.dim)
        for i in range(self.dim):
            F[i] = self.loadF(i)
        return F

    # Used for plotting, computes the y values of FEM solution
    # on this finite element. Returns tuple of transformed x and y
    def xy(self, u):
        x = np.linspace(0, 1, 50)
        y = np.zeros_like(x)
        for i, b in enumerate(self.basis):
            y += b(x) * u[self.nodes[i]]
        trans_x = self.F[0] + self.F[1] * x
        return (trans_x, y)

    # Compute the error to the exact solution sin(x) for this
    # finite element in energy norm
    def get_error(self, u):
        x, w = np.polynomial.legendre.leggauss(self.m + 6)
        # Transformation of Gauss points and weights to (0,1)
        x = (x + 1) / 2
        w /= 2

        FEM_sol = np.zeros_like(x)
        FEM_sol_d = np.zeros_like(x)
        for i in range(len(self.basis)):
            FEM_sol += self.basis[i](x) * u[self.nodes[i]]
            FEM_sol_d += self.dbasis[i](x) * u[self.nodes[i]]

        diff = FEM_sol - np.sin(self.F[0] + self.F[1] * x)
        diff_d = FEM_sol_d / self.h - np.cos(self.F[0] + self.F[1] * x)

        SUM1 = self.h * np.sum(diff_d ** 2 * w)
        SUM2 = self.h * np.sum(diff ** 2 * w)
        return (SUM1 + SUM2)


class FEM:
    def __init__(self, num_FE, p):
        self.gridp = np.linspace(0, np.pi, num_FE + 1)
        self.element = []
        self.num_FE = num_FE
        self.p = p
        self.N = N = num_FE + 1 + (p - 1) * num_FE
        for i in range(num_FE):
            # Affine linear map of reference element (0,1) to FE
            F = (self.gridp[i], self.gridp[i + 1] - self.gridp[i])
            # linear basis function nodes
            nodes = [i, i + 1]
            for j in range(p - 1):
                # p > 2 basis function nodes
                nodes += [num_FE + 1 + j * num_FE + i]

            self.element += [FE(nodes, F, p, N)]

    # Generate system stiffness matrix
    def gen_A(self):
        self.A = np.zeros((self.N, self.N))
        for E in self.element:
            A_FE = E.StiffnessMatrix
            C = E.ConnectivityMatrix
            self.A += C @ A_FE @ C.T
        # To take into account dirichlet bc the respective rows
        # and colums are zeroed except for the diagonal element (=1)
        self.A[0, :] = 0
        self.A[self.num_FE, :] = 0
        self.A[:, 0] = 0
        self.A[:, self.num_FE] = 0
        self.A[0, 0] = 1
        self.A[self.num_FE, self.num_FE] = 1

    # Generate system load vector
    def gen_F(self):
        self.F = np.zeros(self.N)
        for E in self.element:
            F_FE = E.LoadVector
            C = E.ConnectivityMatrix
            self.F += C @ F_FE

        # To take into account dirichlet bc the respective rows
        # are zeroed
        self.F[0] = 0
        self.F[self.num_FE] = 0

    def solve(self):
        self.u = np.linalg.solve(self.A, self.F)

    def plot_solution(self):
        X = np.array([])
        Y = np.array([])
        for E in self.element:
            x, y = E.xy(self.u)
            X = np.append(X, x)
            Y = np.append(Y, y)
        plt.plot(X, Y, label=f"FEM, p={self.p}")

    # Returns the error to exact solution sin(x) in energy norm
    def get_error(self):
        err = 0
        for E in self.element:
            err += E.get_error(self.u)

        return np.sqrt(err)


FEs = np.array([2, 3, 10, 20, 40])
p = np.array([1, 2, 3, 4])
energy_error = np.zeros((len(p), len(FEs)))

for i, num_FE in enumerate(FEs):
    for j, pp in enumerate(p):
        FEM1 = FEM(num_FE, pp)
        FEM1.gen_A()
        FEM1.gen_F()
        FEM1.solve()
        energy_error[j, i] = FEM1.get_error()

plt.figure()
for j, pp in enumerate(p):
    plt.loglog(FEs, energy_error[j, :], ".-", label=f"p={pp}")

plt.loglog(FEs, 1 / FEs, ":", color="black", label=r"$h$")
plt.loglog(FEs, 1 / FEs ** 2, ":", color="black", label=r"$h^2$")
plt.loglog(FEs, 1 / FEs ** 3 * 0.1, ":", color="black", label=r"$h^3$")
plt.loglog(FEs, 1 / FEs ** 4 * 0.01, ":", color="black", label=r"$h^4$")
plt.legend()
plt.xlabel("N")
plt.ylabel("Energy norm error")

# Plot of the solution
plt.figure()

for j, pp in enumerate(p):
    FEM1 = FEM(2, pp)
    FEM1.gen_A()
    FEM1.gen_F()
    FEM1.solve()
    FEM1.plot_solution()

plt.title(r"FEM solutions with $h=\pi/2$")
plt.xlabel("x")
plt.ylabel("y")
x = np.linspace(0, np.pi, 1000)
y = np.sin(x)
plt.plot(x, y, ":", color="black", label="True solution")
plt.legend()
plt.show()