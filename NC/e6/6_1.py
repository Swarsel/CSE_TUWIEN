import matplotlib.pyplot as plt


def f1(n, x0=1):
    # simple division, same as exact
    x = x0
    for _ in range(1,n+1):
        x = x / 3
    return x


def f2(n, x0=1, x1=1 / 3):
    # we stay similar with xn = xtrue + error
    # error = 4/3 error_n - 1/3 error_n-1
    # absolute error approximately constant, which makes relative error bigger
    x_minus_one = x0
    xn = x1
    for _ in range(2,n+1):
        x = (4 / 3) * xn - (1 / 3) * x_minus_one
        x_minus_one, xn = xn, x
    return x


def f3(n, x0=1, x1=1 / 3):
    # we overshoot the true value with xn = xtrue + error
    # error = 10/3 error_n - 3/3 error_n-1
    # error scales with approx (7/3)^n
    x_minus_one = x0
    xn = x1
    for _ in range(2,n+1):
        x = (10 / 3) * xn - x_minus_one
        x_minus_one, xn = xn, x
    return x


def ex(n):
    return (1 / 3) ** n


for delta in [0, 10 ** (-14)]:
    #delta shifts result but error behaves the same
    absolute1 = []
    absolute2 = []
    absolute3 = []
    relative1 = []
    relative2 = []
    relative3 = []
    fig, ax = plt.subplots()
    ax2 = ax.twinx()
    Ns = list(range(2, 100))
    for N in Ns:
        rec1 = f1(N, x0=1 + delta)
        rec2 = f2(N, x0=1 + delta, x1=1 / 3 + delta)
        rec3 = f3(N, x0=1 + delta, x1=1 / 3 + delta)
        exact = ex(N)
        absolute1.append(abs(rec1 - exact))
        absolute2.append(abs(rec2 - exact))
        absolute3.append(abs(rec3 - exact))
        relative1.append(abs((rec1 - exact) / exact))
        relative2.append(abs((rec2 - exact) / exact))
        relative3.append(abs((rec3 - exact) / exact))

    abs1 = ax.semilogy(Ns, absolute1, label='absolute error f1', color="r")
    abs2 = ax.semilogy(Ns, absolute2, label='absolute error f2', color="g")
    abs3 = ax.semilogy(Ns, absolute3, label='absolute error f3', color="b")

    rel1 = ax2.semilogy(Ns, relative1, label='relative error f1', color="r", linestyle="--")
    rel2 = ax2.semilogy(Ns, relative2, label='relative error f2', color="g", linestyle="--")
    rel3 = ax2.semilogy(Ns, relative3, label='relative error f3', color="b", linestyle="--")

    ax.set_ylabel('Absolute error')
    ax2.set_ylabel('Relative error')
    ax.set_title(f'Absolute and relative errors, delta={delta}')
    # ax.legend()
    # ax2.legend()
    fig.legend(loc="upper right", bbox_to_anchor=(1, 1), bbox_transform=ax.transAxes)

    fig.tight_layout()

    plt.show()
