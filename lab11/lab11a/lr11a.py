# F = 2x1^2 + 3x2^2 - 2x1x2 - 8x1 - x2,  x0 = (-4; -4)
# Ограничение: x1 + 2x2 >= 6  ->  g(x) = 6 - x1 - 2x2 <= 0
# Phi_r = F + r*max(0,g)^2, внутри - Флетчер-Ривс (ЛР 9), r0=1, c=10, eps=0.01

import numpy as np
import matplotlib.pyplot as plt

NF = 0   # счётчик вычислений Phi
NG = 0   # счётчик вычислений градиента Phi


def F(x):
    return 2*x[0]**2 + 3*x[1]**2 - 2*x[0]*x[1] - 8*x[0] - x[1]

def gradF(x):
    return np.array([4*x[0] - 2*x[1] - 8, 6*x[1] - 2*x[0] - 1])

def g(x):
    return 6 - x[0] - 2*x[1]

def gradg(x):
    return np.array([-1.0, -2.0])


# штрафная функция и градиент
def Phi(x, r):
    s = max(0.0, g(x))
    return F(x) + r*s*s

def gradPhi(x, r):
    s = max(0.0, g(x))
    return gradF(x) + 2*r*s*gradg(x)    # "пружина" к границе


# одномерный поиск: Свенн + золотое сечение
def swann(f, step=0.01):
    x0 = 0.0
    x1 = step
    f0 = f(x0)
    f1 = f(x1)
    if f1 >= f0:
        return x0, x1
    while True:
        step = step*2
        x2 = x1 + step
        f2 = f(x2)
        if f2 >= f1:
            return x0, x2
        x0 = x1
        x1 = x2
        f1 = f2

def golden(f, a, b, tol=1e-6):
    k = 0.6180339887
    x1 = b - k*(b - a)
    x2 = a + k*(b - a)
    f1 = f(x1)
    f2 = f(x2)
    while b - a > tol:
        if f1 <= f2:
            b = x2
            x2 = x1
            f2 = f1
            x1 = b - k*(b - a)
            f1 = f(x1)
        else:
            a = x1
            x1 = x2
            f1 = f2
            x2 = a + k*(b - a)
            f2 = f(x2)
    return (a + b)/2


# Флетчер-Ривс для Phi_r
def fletcher_reeves(r, x0, eps, verbose=False, max_iter=2000):
    global NF, NG
    x = np.array(x0, float)
    gr = gradPhi(x, r)
    NG += 1
    d = -gr
    path = [x.copy()]
    it = 0
    while np.linalg.norm(gr) > eps and it < max_iter:
        def f(h):
            global NF
            NF += 1
            return Phi(x + h*d, r)
        a, b = swann(f)
        h = golden(f, a, b)
        x = x + h*d
        gnew = gradPhi(x, r)
        NG += 1
        if (it + 1) % 2 == 0:            # рестарт каждые 2 шага
            d = -gnew
        else:
            beta = (gnew @ gnew)/(gr @ gr)
            d = -gnew + beta*d
        gr = gnew
        it += 1
        path.append(x.copy())
        if verbose:
            print("   FR итер %d: h=%.6f  x=(%.6f; %.6f)  |gradPhi|=%.3e"
                  % (it, h, x[0], x[1], np.linalg.norm(gr)))
    if it >= max_iter:
        print("   !!! достигнут лимит итераций внутреннего метода")
    return x, it, path


# внешний цикл
def sumt(x0, r0=1.0, c=10.0, eps=0.01, eps_in=0.01, verbose=True):
    global NF, NG
    NF = 0
    NG = 0
    x = np.array(x0, float)
    r = r0
    rows = []
    traj = [x.copy()]
    allpath = []
    k = 0
    while True:
        k += 1
        nf0 = NF
        ng0 = NG
        x, it, path = fletcher_reeves(r, x, eps_in, verbose=(verbose and k == 1))
        allpath.append(path)
        viol = max(0.0, g(x))
        P = viol**2
        lam = 2*r*viol
        rows.append([k, r, x[0], x[1], F(x), viol, r*P, lam, it, NF - nf0, NG - ng0])
        traj.append(x.copy())
        if r*P <= eps:
            break
        r = r*c
    return x, rows, traj, allpath


def print_table(rows):
    print(" k        r        x1         x2         F         g(x)      r*P     2r*g   итер  NPhi  NgradPhi")
    for w in rows:
        print("%2d %9.0f %10.6f %10.6f %10.5f %10.6f %9.5f %8.4f %4d %5d %5d"
              % (w[0], w[1], w[2], w[3], w[4], w[5], w[6], w[7], w[8], w[9], w[10]))

def x_analytic(r):
    # (4+2r)x1 + (-2+4r)x2 = 8+12r ;  (-2+4r)x1 + (6+8r)x2 = 1+24r
    a11 = 4 + 2*r
    a12 = -2 + 4*r
    a22 = 6 + 8*r
    b1 = 8 + 12*r
    b2 = 1 + 24*r
    det = a11*a22 - a12*a12
    x1 = (b1*a22 - a12*b2)/det
    x2 = (a11*b2 - a12*b1)/det
    return np.array([x1, x2])

x0 = np.array([-4.0, -4.0])

print("=== 1. Точное решение (аналитика) ===")
print("g(x) = 6 - x1 - 2x2 <= 0;  g(x0) = %.1f" % g(x0))
xs = np.array([3.0, 1.5])
lam_exact = 1.0
print("На границе x1 = 6 - 2x2: F(x2) = 15x2^2 - 45x2 + 24 -> x2 = 1.5, x1 = 3")
print("x* = (3; 1.5), F* = %.4f" % F(xs))
print("gradF(x*) = (%.1f; %.1f) = -lambda*(-1; -2)  ->  lambda = %.1f" %
      (gradF(xs)[0], gradF(xs)[1], lam_exact))
xu = np.array([2.5, 1.0])
print("Безусловный минимум (2.5; 1): g = %.1f > 0 - нарушает ограничение\n" % g(xu))

print("=== 2-3. Прогон r0=1, c=10, eps=0.01 (внутри Флетчер-Ривс) ===")
x_res, rows, traj, allpath = sumt(x0)
print()
print_table(rows)
NPhi = NF
NgradPhi = NG
print("\nИТОГО: внешних итераций %d, NPhi = %d, NgradPhi = %d" % (len(rows), NPhi, NgradPhi))
viol = max(0.0, g(x_res))
print("Ответ: x ~ (%.6f; %.6f), F = %.5f (точно %.4f), нарушение = %.2e, lambda ~ %.4f (точно %.1f)"
      % (x_res[0], x_res[1], F(x_res), F(xs), viol, rows[-1][7], lam_exact))

print("\n=== 4. Проверка аналитикой: x(r) из системы 2x2 ===")
print("      r        x1(числ)   x2(числ)   x1(аналит) x2(аналит)")
for w in rows:
    xa = x_analytic(w[1])
    print("%9.0f %10.6f %10.6f %10.6f %10.6f" % (w[1], w[2], w[3], xa[0], xa[1]))
print("Оценка нарушения lambda/(2r): ", ["%.2e" % (lam_exact/(2*w[1])) for w in rows])

print("\n=== 6. Доводка проекцией на границу ===")
gg = gradg(x_res)
x_hat = x_res - g(x_res)*gg/(gg @ gg)
print("x_hat = (%.6f; %.6f), g(x_hat) = %.2e, F(x_hat) = %.6f" % (x_hat[0], x_hat[1], g(x_hat), F(x_hat)))

print("\nСравнение: сразу r = 1e6 со старта (без теплого старта)")
NF = 0
NG = 0
x_big, it_big, _ = fletcher_reeves(1e6, x0, 0.01)
print("x = (%.6f; %.6f), итераций ФР = %d, NPhi = %d, нарушение = %.2e"
      % (x_big[0], x_big[1], it_big, NF, max(0.0, g(x_big))))
print("(для сравнения: r = 1..%.0f с теплым стартом: NPhi = %d)" % (rows[-1][1], NPhi))



# График 1
fig1, ax = plt.subplots(figsize=(8, 7))
X1, X2 = np.meshgrid(np.linspace(-6, 7, 400), np.linspace(-6, 6, 400))
Z = 2*X1**2 + 3*X2**2 - 2*X1*X2 - 8*X1 - X2
ax.contour(X1, X2, Z, levels=[-8, -5, 0, 10, 30, 60, 100], cmap="Blues_r", linewidths=1)
ax.contourf(X1, X2, 6 - X1 - 2*X2, levels=[-100, 0], colors=["#c8f0c8"], alpha=0.5)
t = np.linspace(-6, 7, 10)
ax.plot(t, (6 - t)/2, "g-", lw=2, label="граница x1 + 2x2 = 6 (Omega - справа/выше)")
for p in allpath[:1]:                       # внутренние шаги первой задачи
    p = np.array(p)
    ax.plot(p[:, 0], p[:, 1], "o--", color="orange", ms=4, label="шаги ФР при r = 1")
tr = np.array(traj)
ax.plot(tr[:, 0], tr[:, 1], "r.-", ms=8, label="x(r)")
for i in range(1, len(tr)):
    ax.annotate("r=%g" % rows[i-1][1], tr[i], textcoords="offset points", xytext=(6, -12), fontsize=9)
ax.plot(*x0, "ks", label="старт x0 = (-4; -4)")
ax.plot(2.5, 1.0, "m*", ms=12, label="безусловный минимум (2.5; 1)")
ax.plot(*xs, "g*", ms=14, label="условный минимум x* = (3; 1.5)")
ax.set_xlim(-6, 7)
ax.set_ylim(-6, 6)
ax.set_aspect("equal")
ax.set_xlabel("x1")
ax.set_ylabel("x2")
ax.set_title("Внешний штраф: траектория x(r) подходит к границе снаружи")
ax.legend(loc="lower left", fontsize=8)
ax.grid(alpha=0.3)
fig1.savefig("lr11a_trajectory.png", dpi=130)

# График 2
fig2, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.5))
rr = [w[1] for w in rows]
gv = [w[5] for w in rows]
a1.loglog(rr, gv, "ro-", label="g(x(r)) - расчёт")
a1.loglog(rr, [lam_exact/(2*v) for v in rr], "k--", label="lambda/(2r)")
a1.set_xlabel("r")
a1.set_ylabel("нарушение g(x(r))")
a1.set_title("Нарушение падает как 1/r")
a1.grid(alpha=0.3, which="both")
a1.legend()
a2.semilogx(rr, [w[7] for w in rows], "bo-", label="2r*g(x(r))")
a2.axhline(lam_exact, color="k", ls="--", label="lambda = 1")
a2.set_xlabel("r")
a2.set_ylabel("оценка множителя")
a2.set_title("2r*g(x(r)) -> lambda")
a2.grid(alpha=0.3)
a2.legend()
fig2.tight_layout()
fig2.savefig("lr11a_violation.png", dpi=130)

# График 3
fig3, ax3 = plt.subplots(figsize=(6.5, 4.5))
ax3.semilogx(rr, [w[4] for w in rows], "go-", label="F(x(r))")
ax3.axhline(F(xs), color="k", ls="--", label="F* = %.3f" % F(xs))
ax3.set_xlabel("r")
ax3.set_ylabel("F")
ax3.set_title("F(x(r)) растёт к F* снизу")
ax3.grid(alpha=0.3)
ax3.legend()
fig3.tight_layout()
fig3.savefig("lr11a_F.png", dpi=130)

print("\nГрафики сохранены: lr11a_trajectory.png, lr11a_violation.png, lr11a_F.png")
plt.show()
