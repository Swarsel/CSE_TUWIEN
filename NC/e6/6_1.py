import matplotlib.pyplot as plt

delta = 10 ** (-14)

def f1(n, x0 = 1 + delta):
    x = x0
    for _ in range(n):
        x = x / 3
    return x

def f2(n, x0 = 1  + delta, x1 = 1/ 3  + delta):
    xm = x0
    xn = x1
    for _ in range(n):
        x = (4 / 3) * xn - (1 / 3) * xm
        xm, xn = xn, x
    return x

def f3(n, x0 = 1  + delta, x1 = 1 / 3  + delta):
    xm = x0
    xn = x1
    for _ in range(n):
        x = (10 / 3) * xn - xm
        xm, xn = xn, x
    return x

def ex(n):
    return (1 / 3) ** n

absolute1 = []
absolute2 = []
absolute3 = []
relative1 = []
relative2 = []
relative3 = []
fig, ax = plt.subplots()
ax2 = ax.twinx()
Ns = list(range(1,100))
for N in Ns:
    rec1 = f1(N)
    rec2 = f2(N)
    rec3 = f3(N)
    exact = ex(N)
    absolute1.append(abs(rec1 - exact))
    absolute2.append(abs(rec2 - exact))
    absolute3.append(abs(rec3 - exact))
    relative1.append(abs((rec1 - exact)/rec1))
    relative2.append(abs((rec2 - exact)/rec2))
    relative3.append(abs((rec3 - exact)/rec3))


abs1 = ax.semilogy(Ns, absolute1, label='absolute error f1', color="r")
abs2 = ax.semilogy(Ns, absolute2, label='absolute error f2', color="g")
abs3 = ax.semilogy(Ns, absolute3, label='absolute error f3', color="b")

rel1 = ax2.plot(Ns, relative1, label='relative error f1', color="r", linestyle="--")
rel2 = ax2.plot(Ns, relative2, label='relative error f2', color="g", linestyle="--")
rel3 = ax2.plot(Ns, relative3, label='relative error f3', color="b", linestyle="--")

ax.set_ylabel('Absolute error')
ax2.set_ylabel('Relative error')
ax.set_title('Absolute and relative errors of recursion with perturbed initial values')
ax.legend()
ax2.legend()


fig.tight_layout()

plt.show()
