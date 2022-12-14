import matplotlib.pyplot as plt
import numpy as np
import scipy as sp

n = 10
e = np.ones((n, 1))
b = np.array([1] + [0 for _ in range(n - 1)])
b = np.reshape(b, (n, 1))
A = 10 * np.identity(n) + b @ e.T + e @ b.T
A_tilde = np.flip(A)

# permutation matrix not needed -> _
_, L, U = sp.linalg.lu(A)
_, L_tilde, U_tilde = sp.linalg.lu(A_tilde)

fig, axs = plt.subplots(2, 3)

axs[0, 0].spy(A)
axs[0, 0].set_title('A')
axs[0, 1].spy(L)
axs[0, 1].set_title('L')
axs[0, 2].spy(U)
axs[0, 2].set_title('U')

axs[1, 0].spy(A_tilde)
axs[1, 0].set_title('A_tilde')
axs[1, 1].spy(L_tilde)
axs[1, 1].set_title('L_tilde')
axs[1, 2].spy(U_tilde)
axs[1, 2].set_title('U_tilde')

plt.show()
