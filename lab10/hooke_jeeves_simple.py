import math
import numpy as np
import matplotlib.pyplot as plt

nF = 0


def F4(p):
    global nF
    nF += 1
    x1, x2 = p
    return 2*x1**2 + 3*x2**2 - 2*x1*x2 - 8*x1 - x2

def G(p):                       # негладкая функция
    global nF
    nF += 1
    x1, x2 = p
    return 2*abs(x1 - 1) + 3*abs(x2 + 2) + 0.5*(x1 - 1)**2

def Rosen(p):                   # Розенброк
    global nF
    nF += 1
    x1, x2 = p
    return (1 - x1)**2 + 100*(x2 - x1**2)**2

def explore(F, y, Fy, delta):
    z = y[:]
    Fz = Fy
    for i in range(2):
        for sign in (1, -1):
            t = z[:]
            t[i] += sign * delta
            Ft = F(t)
            if Ft < Fz:
                z = t
                Fz = Ft
                break
    return z, Fz

def pattern(F, z, base):
    y = [2*z[0] - base[0], 2*z[1] - base[1]]
    return y, F(y)


def pt(p):
    return "(%.4f; %.4f)" % (p[0], p[1])


# Хук-Дживс
def hooke_jeeves(F, x0, delta0, eps, verbose):
    global nF
    nF = 0
    base = x0[:]
    Fb = F(base)
    y, Fy = base[:], Fb
    delta = delta0
    from_pattern = False
    k = 0
    bases = [base]
    base_F = [Fb]

    while True:
        n_before = nF
        z, Fz = explore(F, y, Fy, delta)
        probes = nF - n_before
        if verbose:
            kind = "из образца" if from_pattern else "из базы"
            print(k, "исследующий", kind, pt(y), "->", pt(z),
                  "F(z) = %.6f" % Fz, "проб:", probes, "N_F:", nF)

        if Fz < Fb:
            y, Fy = pattern(F, z, base)
            base, Fb = z, Fz
            from_pattern = True
            k += 1
            bases.append(base)
            base_F.append(Fb)
            if verbose:
                print(k, "новая база", pt(base), "F = %.6f" % Fb,
                      "-> образец", pt(y), "F = %.6f" % Fy, "N_F:", nF)
        elif from_pattern:
            y, Fy = base[:], Fb
            from_pattern = False
            if verbose:
                print(k, "возврат в базу", pt(base))
        else:
            delta = delta / 2
            if verbose:
                print(k, "дробление: delta =", delta)
            if delta < eps:
                break

    return base, Fb, k, nF, bases, base_F


def run(title, F, x0, x_exact, delta0, eps, verbose=False):
    print("=" * 80)
    print(title, "| старт", pt(x0), "| delta0 =", delta0, "| eps =", eps)
    print("=" * 80)
    x, Fx, k, n, bases, base_F = hooke_jeeves(F, x0, delta0, eps, verbose)
    print("Базовые точки:")
    for i in range(len(bases)):
        print("  ", i, pt(bases[i]), "F = %.6f" % base_F[i])
    err = math.hypot(x[0] - x_exact[0], x[1] - x_exact[1])
    print("x* =", pt(x), " F(x*) = %.3e" % Fx)
    print("||x* - x_точн|| = %.2e" % err)
    print("Итог: баз =", k, " N_F =", n, " N_grad = 0\n")
    return bases


# графики
def draw(F, runs, xlim, ylim, title, filename):
    a = np.linspace(xlim[0], xlim[1], 300)
    b = np.linspace(ylim[0], ylim[1], 300)
    X, Y = np.meshgrid(a, b)
    Z = F([X, Y])
    plt.figure(figsize=(8, 6))
    plt.contour(X, Y, Z, levels=30, colors="gray", linewidths=0.6)
    for label, bases in runs:
        xs = [p[0] for p in bases]
        ys = [p[1] for p in bases]
        plt.plot(xs, ys, "o-", label=label)
    plt.title(title)
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.savefig(filename, dpi=150)
    plt.show()


exact4 = [2.5, 1.0]
start4 = [-4.0, -4.0]

b1 = run("Вариант 4", F4, start4, exact4, 1.0, 0.01, verbose=True)
b2 = run("Вариант 4", F4, start4, exact4, 0.7, 0.01)

bg = run("Негладкая функция G", G, [5.0, -3.0], [1.0, -2.0], 1.0, 0.01, verbose=True)

run("Розенброк", Rosen, [-1.2, 1.0], [1.0, 1.0], 1.0, 0.01)
run("Розенброк", Rosen, [-1.2, 1.0], [1.0, 1.0], 1.0, 1e-4)
run("Розенброк", Rosen, [-1.2, 1.0], [1.0, 1.0], 0.1, 0.01)

draw(F4, [("delta0 = 1", b1), ("delta0 = 0.7", b2)],
     (-5, 5), (-5, 4), "Вариант 4: траектория баз", "hj_variant4.png")
draw(G, [("delta0 = 1", bg)],
     (-2, 6), (-4, 1), "Негладкая функция G", "hj_nonsmooth.png")
