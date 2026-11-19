# ЛР 11, часть Б: эксперименты со штрафами и барьером, вариант 4
# F = 2x1^2 + 3x2^2 - 2x1x2 - 8x1 - x2,  x0 = (-4; -4)
# Ограничение 1: x1 + 2x2 >= 6   -> g1 = 6 - x1 - 2x2 <= 0
# Ограничение 2: x2 <= 1         -> g2 = x2 - 1 <= 0
# Круг:          x1^2 + x2^2 >= 16 -> g3 = 16 - x1^2 - x2^2 <= 0
# Параметры: r0 = 1, c = 10, eps = 0.01, внутри Флетчер-Ривс с |grad| <= 0.01
import numpy as np
import matplotlib.pyplot as plt

NF = 0   # счётчик вычислений Phi
NG = 0   # счётчик вычислений градиента


# ---------- исходная функция ----------
def F(x):
    return 2*x[0]**2 + 3*x[1]**2 - 2*x[0]*x[1] - 8*x[0] - x[1]

def gradF(x):
    return np.array([4*x[0] - 2*x[1] - 8, 6*x[1] - 2*x[0] - 1])


# ---------- ограничения: список [g, grad_g, тип]; тип 0 - неравенство, 1 - равенство ----------
def g1(x):
    return 6 - x[0] - 2*x[1]
def gg1(x):
    return np.array([-1.0, -2.0])

def g2(x):
    return x[1] - 1
def gg2(x):
    return np.array([0.0, 1.0])

def g3(x):
    return 16 - x[0]**2 - x[1]**2
def gg3(x):
    return np.array([-2*x[0], -2*x[1]])

C1 = [[g1, gg1, 0]]
C1_EQ = [[g1, gg1, 1]]
C12 = [[g1, gg1, 0], [g2, gg2, 0]]
C_CIRCLE = [[g3, gg3, 0]]


# ---------- внешний штраф ----------
def viol(v, kind):
    if kind == 0:
        return max(0.0, v)      # зрезка для неравенства
    return v                    # для равенства зрезка не нужна

def Phi_pen(x, r, cons):
    P = 0.0
    for g, gg, kind in cons:
        s = viol(g(x), kind)
        P = P + s*s
    return F(x) + r*P

def grad_pen(x, r, cons):
    v = gradF(x)
    for g, gg, kind in cons:
        s = viol(g(x), kind)
        v = v + 2*r*s*gg(x)
    return v

def penalty_P(x, cons):
    P = 0.0
    for g, gg, kind in cons:
        s = viol(g(x), kind)
        P = P + s*s
    return P


# ---------- внутренний штраф (логарифмический барьер) ----------
def Phi_bar(x, mu, cons):
    s = 0.0
    for g, gg, kind in cons:
        v = g(x)
        if v >= 0:
            return float("inf")      # стена: снаружи +inf
        s = s - mu*np.log(-v)
    return F(x) + s

def grad_bar(x, mu, cons):
    v = gradF(x)
    for g, gg, kind in cons:
        v = v - mu/g(x)*gg(x)
    return v


# ---------- одномерный поиск: Свенн + золотое сечение ----------
def swann(f, step=0.01):
    x0 = 0.0
    x1 = step
    f0 = f(x0)
    f1 = f(x1)
    if f1 >= f0:                     # inf тоже считается ростом
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
        if f1 <= f2:                 # при f1 = f2 = inf идём влево, от стены
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


# ---------- внутренние методы ----------
def fletcher_reeves(phi, grad, x0, eps, max_iter=2000):
    global NF, NG
    x = np.array(x0, float)
    gr = grad(x)
    NG += 1
    d = -gr
    it = 0
    path = [x.copy()]
    while np.linalg.norm(gr) > eps and it < max_iter:
        def f(h):
            global NF
            NF += 1
            return phi(x + h*d)
        a, b = swann(f)
        h = golden(f, a, b)
        x = x + h*d
        gnew = grad(x)
        NG += 1
        if (it + 1) % 2 == 0:        # рестарт каждые n = 2 шага
            d = -gnew
        else:
            beta = (gnew @ gnew)/(gr @ gr)
            d = -gnew + beta*d
        gr = gnew
        it += 1
        path.append(x.copy())
    if it >= max_iter:
        print("   !!! лимит итераций внутреннего метода")
    return x, it, path

def sphere_descent(phi, grad, x0, eps, h0=1.0, max_iter=20000):
    # метод ЛР 8, метрика "sphere": d = -g/||g||, шаг h дробится пополам,
    # пока Phi не уменьшится; h между итерациями НЕ сбрасывается (как в gradient_fixed.cpp)
    global NF, NG
    x = np.array(x0, float)
    h = h0
    fx = phi(x)
    NF += 1
    it = 0
    stuck = False
    while it < max_iter:
        gr = grad(x)
        NG += 1
        ng = np.linalg.norm(gr)
        if ng <= eps:
            break
        d = -gr/ng
        accepted = False
        while True:
            xn = x + h*d
            fn = phi(xn)
            NF += 1
            if fn < fx:
                accepted = True
                break
            h = h/2
            if h < 1e-10:
                break
        if not accepted:
            stuck = True
            break
        x = xn
        fx = fn
        it += 1
    if stuck:
        print("   !!! шаг раздробился до h_min, метод остановлен (||grad|| = %.3e)" % ng)
    elif it >= max_iter:
        print("   !!! лимит итераций: сферическая метрика не сошлась")
    return x, it, [x.copy()]


# ---------- внешний цикл: штраф ----------
def sumt_pen(cons, x0, r0=1.0, c=10.0, eps=0.01, eps_in=0.01, inner="fr"):
    global NF, NG
    NF = 0
    NG = 0
    x = np.array(x0, float)
    r = r0
    rows = []
    traj = [x.copy()]
    k = 0
    while True:
        k = k + 1
        nf0 = NF
        ng0 = NG
        phi = lambda z: Phi_pen(z, r, cons)
        grd = lambda z: grad_pen(z, r, cons)
        if inner == "fr":
            x, it, path = fletcher_reeves(phi, grd, x, eps_in)
        else:
            x, it, path = sphere_descent(phi, grd, x, eps_in)
        vs = [viol(g(x), kind) for g, gg, kind in cons]
        lams = [2*r*v for v in vs]
        rP = r*penalty_P(x, cons)
        rows.append([k, r, x.copy(), F(x), vs, rP, lams, it, NF - nf0, NG - ng0])
        traj.append(x.copy())
        if rP <= eps or r > 1e12:
            break
        r = r*c
    return x, rows, traj


# ---------- внешний цикл: барьер ----------
def sumt_bar(cons, x0, mu0=10.0, c=10.0, eps=0.01):
    global NF, NG
    NF = 0
    NG = 0
    x = np.array(x0, float)
    for g, gg, kind in cons:
        if g(x) >= 0:
            print("!!! старт не строго внутри области")
            return x, [], [x.copy()]
    mu = mu0
    rows = []
    traj = [x.copy()]
    k = 0
    while True:
        k = k + 1
        nf0 = NF
        ng0 = NG
        phi = lambda z: Phi_bar(z, mu, cons)
        grd = lambda z: grad_bar(z, mu, cons)
        x, it, path = fletcher_reeves(phi, grd, x, 0.01)
        gs = [g(x) for g, gg, kind in cons]
        lams = [-mu/v for v in gs]
        rows.append([k, mu, x.copy(), F(x), gs, mu*len(cons), lams, it, NF - nf0, NG - ng0])
        traj.append(x.copy())
        if mu*len(cons) <= eps:
            break
        mu = mu/c
    return x, rows, traj


# ---------- печать ----------
def fl(v, f="%.5f"):
    return "(" + "; ".join(f % t for t in v) + ")"

def print_pen(rows):
    print(" k        r      x(r)                      F        нарушения            r*P      оценки lambda    ит  NPhi NgrPhi")
    for w in rows:
        print("%2d %9.0f %-24s %10.5f %-20s %9.5f %-16s %3d %5d %5d"
              % (w[0], w[1], fl(w[2], "%.5f"), w[3], fl(w[4], "%.5f"), w[5], fl(w[6], "%.3f"), w[7], w[8], w[9]))

def print_bar(rows, xstar_F):
    print(" k       mu     x(mu)                      g(x)               F        F-F*      mu/(-g)         ит  NPhi NgrPhi")
    for w in rows:
        print("%2d %8.4f %-24s %-18s %10.5f %9.5f %-14s %3d %5d %5d"
              % (w[0], w[1], fl(w[2], "%.5f"), fl(w[4], "%.5f"), w[3], w[3] - xstar_F, fl(w[6], "%.3f"), w[7], w[8], w[9]))

def totals(rows):
    nf = 0
    ng = 0
    for w in rows:
        nf = nf + w[8]
        ng = ng + w[9]
    return nf, ng


# ======================= ОСНОВНАЯ ПРОГРАММА =======================
x0 = np.array([-4.0, -4.0])
xs1 = np.array([3.0, 1.5])      # точное решение с ограничением 1: F* = -9.75, lambda = 1
Fs1 = F(xs1)
xs12 = np.array([4.0, 1.0])     # точное решение с ограничениями 1+2: F* = -6, lambda = (6; 15)
summary = []                    # сводная таблица (задание 8)

print("Точные решения для контроля:")
print(" только огр.1:    x* = (3; 1.5), F* = %.4f, lambda = 1" % Fs1)
print(" огр.1 + огр.2:   x* = (4; 1),   F* = %.4f, lambda = (6; 15)" % F(xs12))
print(" (в (4; 1): gradF = (6; -3); проверка Куна-Таккера - в пункте 5)\n")

# ---------- 1. Множитель c ----------
print("=" * 70)
print("1. Влияние множителя c  (r0 = 1, eps = 0.01)")
print("=" * 70)
cs = [2, 4, 10, 100]
c_res = []
base_rows = None
base_traj = None
for c in cs:
    x, rows, traj = sumt_pen(C1, x0, c=c)
    nf, ng = totals(rows)
    c_res.append([c, len(rows), rows[-1][1], nf, ng, rows[-1][4][0], x])
    if c == 10:
        base_rows = rows
        base_traj = traj
print("   c  внеш.ит  r_кон     NPhi  NgrPhi  нарушение   x")
for w in c_res:
    print("%4d %6d %9.0f %8d %6d %11.2e  %s" % (w[0], w[1], w[2], w[3], w[4], w[5], fl(w[6], "%.5f")))
print("\nПротокол для базового прогона c = 10:")
print_pen(base_rows)
nf, ng = totals(base_rows)
summary.append(["штраф, неравенство 1 (c=10)", len(base_rows), nf, ng, base_rows[-1][2],
                base_rows[-1][4][0], base_rows[-1][3], base_rows[-1][6][0]])
F_pen = base_rows[-1][3]      # F(x(r)) снизу для двусторонней оценки

# ---------- 2. Внутренний eps ----------
print("\n" + "=" * 70)
print("2. Влияние внутреннего eps  (c = 10)")
print("=" * 70)
eps_list = [0.1, 0.01, 0.001]
e_res = []
for e in eps_list:
    x, rows, traj = sumt_pen(C1, x0, eps_in=e)
    nf, ng = totals(rows)
    e_res.append([e, len(rows), nf, ng, np.linalg.norm(x - xs1), rows[-1][4][0], x])
print(" eps_вн  внеш.ит  NPhi  NgrPhi  |x - x*|    нарушение   x")
for w in e_res:
    print("%7.3f %6d %7d %6d %10.5f %11.2e  %s" % (w[0], w[1], w[2], w[3], w[4], w[5], fl(w[6], "%.5f")))
print("Вывод: нарушение задаётся r, а положение вдоль границы - внутренним eps.")

# ---------- 3. Ограничение 1 как равенство ----------
print("\n" + "=" * 70)
print("3. Ограничение 1 как равенство h(x) = 6 - x1 - 2x2 = 0  (штраф r*h^2, без зрезки)")
print("=" * 70)
x, rows_eq, traj_eq = sumt_pen(C1_EQ, x0)
print_pen(rows_eq)
nf, ng = totals(rows_eq)
print("Итого: %d внешних итераций, NPhi = %d, NgrPhi = %d" % (len(rows_eq), nf, ng))
print("Сравнение с неравенством: x(r) на последнем шаге: нер. %s, рав. %s"
      % (fl(base_rows[-1][2], "%.5f"), fl(rows_eq[-1][2], "%.5f")))
summary.append(["штраф, равенство 1", len(rows_eq), nf, ng, rows_eq[-1][2],
                abs(rows_eq[-1][4][0]), rows_eq[-1][3], rows_eq[-1][6][0]])

# ---------- 4. Логарифмический барьер ----------
print("\n" + "=" * 70)
print("4. Логарифмический барьер, старт строго внутри: (4; 3), g1 = %.1f" % g1(np.array([4.0, 3.0])))
print("=" * 70)
xb0 = np.array([4.0, 3.0])
x_bar, rows_bar, traj_bar = sumt_bar(C1, xb0)
print_bar(rows_bar, Fs1)
nf, ng = totals(rows_bar)
print("Итого: %d внешних итераций, NPhi = %d, NgrPhi = %d" % (len(rows_bar), nf, ng))
F_bar = rows_bar[-1][3]
print("Двусторонняя оценка: F(x(r)) = %.5f  <=  F* <=  F(x(mu)) = %.5f   (точно F* = %.5f)"
      % (F_pen, F_bar, Fs1))
summary.append(["барьер (лог), огр.1", len(rows_bar), nf, ng, rows_bar[-1][2],
                max(0.0, rows_bar[-1][4][0]), F_bar, rows_bar[-1][6][0]])

# ---------- 5. Два ограничения ----------
print("\n" + "=" * 70)
print("5. Ограничения 1 + 2 вместе")
print("=" * 70)
print("Условия Куна-Таккера в (4; 1): gradF = (6; -3) = -l1*(-1; -2) - l2*(0; 1)")
print("  6 = l1  ->  l1 = 6;   -3 = 2*l1 - l2  ->  l2 = 15;  оба >= 0, оба ограничения активны")
x, rows_12, traj_12 = sumt_pen(C12, x0)
print_pen(rows_12)
nf, ng = totals(rows_12)
print("Итого: %d внешних итераций, NPhi = %d, NgrPhi = %d" % (len(rows_12), nf, ng))
print("Точно: x* = (4; 1), F* = %.4f, lambda = (6; 15)" % F(xs12))
summary.append(["штраф, огр.1 + огр.2", len(rows_12), nf, ng, rows_12[-1][2],
                max(rows_12[-1][4]), rows_12[-1][3], rows_12[-1][6][0]])

# ---------- 6. Круг из двух стартов ----------
print("\n" + "=" * 70)
print("6. Область вне круга x1^2 + x2^2 >= 16, два старта")
print("=" * 70)
print("Точные локальные минимумы на окружности (перебором угла):")
print("  A = (3.6513; 1.6333), F = -8.1039 (глобальный);  B = (-2.5119; -3.1130), F = 49.2597")
starts = [np.array([-4.0, -4.0]), np.array([5.0, 3.0])]
circ_res = []
for s in starts:
    print("\nСтарт %s, g(старт) = %.2f" % (fl(s, "%.1f"), g3(s)))
    x, rows, traj = sumt_pen(C_CIRCLE, s)
    print_pen(rows)
    nf, ng = totals(rows)
    print("-> x = %s, |x| = %.5f, F = %.5f, NPhi = %d" % (fl(x, "%.5f"), np.linalg.norm(x), F(x), nf))
    circ_res.append([s, x, traj, rows])
    summary.append(["круг, старт " + fl(s, "%.0f"), len(rows), nf, ng, x,
                    max(0.0, g3(x)), F(x), rows[-1][6][0]])

# ---------- 7. Внутренний метод: сфера против Флетчера-Ривса при r = 100 ----------
print("\n" + "=" * 70)
print("7. Внутренний метод при r = 100 (задача с огр.1, старт (-4; -4))")
print("=" * 70)
NF = 0
NG = 0
r100 = 100.0
x_fr, it_fr, _ = fletcher_reeves(lambda z: Phi_pen(z, r100, C1), lambda z: grad_pen(z, r100, C1), x0, 0.01)
nf_fr = NF
NF = 0
NG = 0
x_sp, it_sp, _ = sphere_descent(lambda z: Phi_pen(z, r100, C1), lambda z: grad_pen(z, r100, C1), x0, 0.01)
nf_sp = NF
print("Флетчер-Ривс:        %4d итераций, NPhi = %6d, x = %s" % (it_fr, nf_fr, fl(x_fr, "%.5f")))
print("Сферическая метрика: %4d итераций, NPhi = %6d, x = %s" % (it_sp, nf_sp, fl(x_sp, "%.5f")))

# ---------- 8. Сводная таблица ----------
print("\n" + "=" * 70)
print("8. Сводная таблица")
print("=" * 70)
print("%-30s %5s %6s %6s  %-24s %10s %10s %8s" % ("прогон", "внеш", "NPhi", "NgrPhi", "ответ", "нарушение", "F", "lambda1"))
for w in summary:
    print("%-30s %5d %6d %6d  %-24s %10.2e %10.5f %8.3f"
          % (w[0], w[1], w[2], w[3], fl(w[4], "%.4f"), w[5], w[6], w[7]))


# ======================= ГРАФИКИ =======================
def base_axes(ax, xlim, ylim, levels):
    X1, X2 = np.meshgrid(np.linspace(xlim[0], xlim[1], 400), np.linspace(ylim[0], ylim[1], 400))
    Z = 2*X1**2 + 3*X2**2 - 2*X1*X2 - 8*X1 - X2
    ax.contour(X1, X2, Z, levels=levels, cmap="Blues_r", linewidths=1)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_aspect("equal")
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    ax.grid(alpha=0.3)
    return X1, X2

# --- График 1: внешний штраф и барьер, ограничение 1 ---
fig1, ax = plt.subplots(figsize=(8, 7))
X1, X2 = base_axes(ax, (-6, 7), (-6, 6), [-9, -5, 0, 10, 30, 60, 100])
ax.contourf(X1, X2, g1(np.array([X1, X2])), levels=[-100, 0], colors=["#c8f0c8"], alpha=0.5)
t = np.linspace(-6, 7, 10)
ax.plot(t, (6 - t)/2, "g-", lw=2, label="x1 + 2x2 = 6")
tr = np.array(base_traj)
ax.plot(tr[:, 0], tr[:, 1], "r.-", ms=8, label="внешний штраф x(r), снаружи")
for i in range(1, len(tr)):
    ax.annotate("r=%g" % base_rows[i-1][1], tr[i], textcoords="offset points", xytext=(6, -12), fontsize=8)
tb = np.array(traj_bar)
ax.plot(tb[:, 0], tb[:, 1], "m.-", ms=8, label="барьер x(mu), изнутри")
for i in range(1, len(tb)):
    ax.annotate("mu=%g" % rows_bar[i-1][1], tb[i], textcoords="offset points", xytext=(6, 4), fontsize=8)
ax.plot(*x0, "ks", label="старт штрафа (-4; -4)")
ax.plot(*xb0, "k^", label="старт барьера (4; 3)")
ax.plot(*xs1, "g*", ms=14, label="x* = (3; 1.5)")
ax.set_title("Два пути к x*: штраф снаружи, барьер изнутри")
ax.legend(loc="lower left", fontsize=8)
fig1.savefig("lr11b_penalty_vs_barrier.png", dpi=130)

# --- График 2: два ограничения ---
fig2, ax = plt.subplots(figsize=(8, 7))
X1, X2 = base_axes(ax, (-6, 8), (-6, 6), [-9, -6, 0, 10, 30, 60, 100])
ok = np.maximum(g1(np.array([X1, X2])), g2(np.array([X1, X2])))
ax.contourf(X1, X2, ok, levels=[-100, 0], colors=["#c8f0c8"], alpha=0.5)
t = np.linspace(-6, 8, 10)
ax.plot(t, (6 - t)/2, "g-", lw=2, label="x1 + 2x2 = 6")
ax.axhline(1, color="orange", lw=2, label="x2 = 1")
tr = np.array(traj_12)
ax.plot(tr[:, 0], tr[:, 1], "r.-", ms=8, label="x(r)")
for i in range(1, len(tr)):
    ax.annotate("r=%g" % rows_12[i-1][1], tr[i], textcoords="offset points", xytext=(6, -12), fontsize=8)
ax.plot(*x0, "ks", label="старт")
ax.plot(*xs1, "b^", label="только огр.1: (3; 1.5)")
ax.plot(*xs12, "g*", ms=14, label="x* = (4; 1), оба активны")
ax.set_title("Два ограничения: ответ в углу области Omega")
ax.legend(loc="lower left", fontsize=8)
fig2.savefig("lr11b_two_constraints.png", dpi=130)

# --- График 3: круг, два старта ---
fig3, ax = plt.subplots(figsize=(8, 7))
X1, X2 = base_axes(ax, (-7, 7), (-7, 7), [-8, 0, 20, 49.26, 80, 150])
a = np.linspace(0, 2*np.pi, 300)
ax.plot(4*np.cos(a), 4*np.sin(a), "g-", lw=2, label="окружность |x| = 4 (Omega - снаружи)")
colors = ["r", "m"]
for i in range(2):
    trc = np.array(circ_res[i][2])
    ax.plot(trc[:, 0], trc[:, 1], ".-", color=colors[i], ms=8, label="старт %s" % fl(starts[i], "%.0f"))
    ax.plot(*starts[i], "s", color=colors[i])
ax.plot(3.6513, 1.6333, "g*", ms=14, label="A: F = -8.104 (глобальный)")
ax.plot(-2.5119, -3.1130, "y*", ms=14, mec="k", label="B: F = 49.26 (локальный)")
ax.plot(2.5, 1.0, "kx", ms=8, label="безусловный минимум")
ax.set_title("Круг: невыпуклая область, два локальных минимума")
ax.legend(loc="upper left", fontsize=8)
fig3.savefig("lr11b_circle.png", dpi=130)

# --- График 4: стоимость при разных c и eps_вн ---
fig4, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.5))
a1.bar([str(w[0]) for w in c_res], [w[3] for w in c_res], color="steelblue")
a1.set_xlabel("множитель c")
a1.set_ylabel("NPhi (всего)")
a1.set_title("Стоимость в зависимости от c")
a1.grid(alpha=0.3, axis="y")
a2.bar([str(w[0]) for w in e_res], [w[4] for w in e_res], color="indianred")
a2.set_xlabel("внутренний eps")
a2.set_ylabel("|x - x*|")
a2.set_title("Точность вдоль границы зависит от eps_вн")
a2.grid(alpha=0.3, axis="y")
fig4.tight_layout()
fig4.savefig("lr11b_c_and_eps.png", dpi=130)

# --- График 5: нарушение g(x(r)), неравенство и равенство ---
fig5, ax5 = plt.subplots(figsize=(6.5, 4.5))
ax5.loglog([w[1] for w in base_rows], [w[4][0] for w in base_rows], "ro-", label="неравенство")
ax5.loglog([w[1] for w in rows_eq], [abs(w[4][0]) for w in rows_eq], "bx--", label="равенство")
ax5.loglog([w[1] for w in base_rows], [1/(2*w[1]) for w in base_rows], "k:", label="lambda/(2r)")
ax5.set_xlabel("r")
ax5.set_ylabel("нарушение")
ax5.set_title("Нарушение при r -> inf")
ax5.grid(alpha=0.3, which="both")
ax5.legend()
fig5.tight_layout()
fig5.savefig("lr11b_violation.png", dpi=130)

print("\nГрафики сохранены: lr11b_*.png (5 файлов)")
plt.show()
