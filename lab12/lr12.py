# ЛР 12: метод проекции градиента (Розен), вариант 4
# F = 2x1^2 + 3x2^2 - 2x1x2 - 8x1 - x2, x0 = (-4; -4)
# Стены в форме a^T x <= b (нормаль a смотрит наружу из Omega):
#   стена 1: x1 + 2x2 >= 6 -> a1 = (-1; -2), b1 = -6
#   стена 2: x2 <= 1-> a2 = (0; 1), b2 = 1
#   стена 3: x2 >= 0-> a3 = (0; -1), b3 = 0 (прогон 3, старт в углу (6; 0))
# eps = 0.01 по ||d||; линейный поиск: Свенн (шаг 0.01) + золотое сечение до 1e-6 с парканом hmax

import numpy as np
import matplotlib.pyplot as plt

NF = 0   # вычисления F
NG = 0   # вычисления градиента


# функция
def F(x):
    return 2*x[0]**2 + 3*x[1]**2 - 2*x[0]*x[1] - 8*x[0] - x[1]

def gradF(x):
    return np.array([4*x[0] - 2*x[1] - 8, 6*x[1] - 2*x[0] - 1])


# линейный поиск: Свенн + золотое сечение
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

def line_search(x, d, hmax):
    global NF
    def phi(h):
        global NF
        NF += 1
        if h > hmax:
            return float("inf")
        return F(x + h*d)
    a, b = swann(phi)
    h = golden(phi, a, b)
    if hmax < float("inf") and hmax - h < 1e-5:
        h = hmax
    return h


# шаг 0: проекция точки на Omega
def feasible(A, b, x):
    for j in range(len(b)):
        if A[j] @ x - b[j] > 1e-9:
            return False
    return True

def project_to_omega(A, b, x):
    if feasible(A, b, x):
        return x.copy()
    cand = []
    for j in range(len(b)):              # проекция на одну стену
        g = A[j] @ x - b[j]
        p = x - g/(A[j] @ A[j])*A[j]
        if feasible(A, b, p):
            cand.append(p)
    for j in range(len(b)):              # углы
        for k in range(j + 1, len(b)):
            M = np.array([A[j], A[k]])
            if abs(np.linalg.det(M)) > 1e-12:
                p = np.linalg.solve(M, np.array([b[j], b[k]]))
                if feasible(A, b, p):
                    cand.append(p)
    best = cand[0]
    for p in cand:
        if np.linalg.norm(p - x) < np.linalg.norm(best - x):
            best = p
    return best


# проектор и множители
def projector(A, I):
    n = A.shape[1]
    if len(I) == 0:
        return np.eye(n)
    AI = A[I]
    return np.eye(n) - AI.T @ np.linalg.solve(AI @ AI.T, AI)

def multipliers(A, I, gr):
    AI = A[I]
    return -np.linalg.solve(AI @ AI.T, AI @ gr)


# метод Розена
def rosen(A, b, x0, ids, eps=0.01, verbose=True, max_iter=200):
    # ids - реальные номера стен
    global NF, NG
    NF = 0
    NG = 0
    x = project_to_omega(A, b, np.array(x0, float))
    if verbose:
        if np.linalg.norm(x - x0) > 1e-12:
            print("Шаг 0: старт недопустим -> проекция на Omega: (%.4f; %.4f) -> (%.4f; %.4f)"
                  % (x0[0], x0[1], x[0], x[1]))
        else:
            print("Шаг 0: старт допустим, x0 = (%.4f; %.4f)" % (x[0], x[1]))
    I = [j for j in range(len(b)) if abs(A[j] @ x - b[j]) <= 1e-9]
    path = [x.copy()]
    rows = []
    if verbose:
        print(" k   x(k)                 I        d                  hmax      h         F         NF")
    k = 0
    u = np.array([])
    while k < max_iter:
        gr = gradF(x)
        NG += 1
        done = False
        while True:                                  # d = 0 -> множители
            P = projector(A, I)
            d = -P @ gr
            if np.linalg.norm(d) > eps:
                break
            if len(I) == 0:
                done = True
                break
            u = multipliers(A, I, gr)
            if verbose:
                print("   d ~ 0, I = %s, u = %s" % (str([ids[j] for j in I]), str(np.round(u, 4))))
            if (u >= -1e-12).all():
                done = True
                break
            j = int(np.argmin(u))
            if verbose:
                print("   u < 0 у стены %d -> она мешает, отпускаем" % ids[I[j]])
            I.pop(j)
        if done:
            break
        hmax = float("inf")                          # паркан
        jhit = -1
        for j in range(len(b)):
            if j not in I and A[j] @ d > 1e-12:
                hj = (b[j] - A[j] @ x)/(A[j] @ d)
                if hj < hmax:
                    hmax = hj
                    jhit = j
        h = line_search(x, d, hmax)
        Iold = [ids[j] for j in I]
        x = x + h*d
        if jhit >= 0 and abs(h - hmax) < 1e-12:
            I.append(jhit)                           # упёрлись в стену
        path.append(x.copy())
        rows.append([k, x.copy(), Iold, d.copy(), hmax, h, F(x), NF])
        if verbose:
            hs = "inf" if hmax == float("inf") else "%.5f" % hmax
            print("%2d  (%8.4f;%8.4f)  %-8s (%8.3f;%8.3f)  %-8s %.6f  %9.5f  %4d"
                  % (k, x[0], x[1], str(Iold), d[0], d[1], hs, h, F(x), NF))
        k = k + 1
    uw = np.zeros(len(b))                            # множители по номерам стен
    if len(I) > 0:
        u = multipliers(A, I, gradF(x))
        for t in range(len(I)):
            uw[I[t]] = u[t]
    return x, u, I, path, k, uw


def check_kt(A, I, x, u):
    # проверка gradF = -A_I^T u
    AI = A[I]
    left = gradF(x)
    right = -AI.T @ u
    return left, right


# метод проекции точки на круг
def circle_projection(x0, R, h=0.1, eps=0.01, max_iter=5000):
    global NG
    NG = 0
    x = np.array(x0, float)
    path = [x.copy()]
    for k in range(max_iter):
        y = x - h*gradF(x)
        NG += 1
        ny = np.linalg.norm(y)
        if ny < R:
            y = y*R/ny
        step = np.linalg.norm(y - x)
        x = y
        path.append(x.copy())
        if step <= eps:
            return x, k + 1, path
    print("   !!! лимит итераций")
    return x, max_iter, path


# ОСНОВНАЯ ПРОГРАММА
x0 = np.array([-4.0, -4.0])
x0in = np.array([12.0, 0.0])          # второй старт внутри Omega
xcorner = np.array([6.0, 0.0])        # старт в углу для прогона 3
A = np.array([[-1.0, -2.0], [0.0, 1.0], [0.0, -1.0]])
b = np.array([-6.0, 1.0, 0.0])

print("Точные решения из ЛР 11: стена 1: x* = (3; 1.5), F* = -9.75, lambda = 1")
print("                         стены 1+2: x* = (4; 1), F* = -6, lambda = (6; 15)\n")

# прогон 1
print("=" * 78)
print("ПРОГОН 1: стена 1, старт x0 = (-4; -4)")
print("=" * 78)
A1 = A[[0]]
b1 = b[[0]]
x1, u1, I1, path1, n1, uw1 = rosen(A1, b1, x0, [1])
nf1 = NF
ng1 = NG
print("Ответ: x* = (%.6f; %.6f), F = %.6f, u = %s, шагов %d, NF = %d, NgradF = %d"
      % (x1[0], x1[1], F(x1), str(np.round(uw1, 6)), n1, nf1, ng1))
l, r = check_kt(A1, I1, x1, u1)
print("Проверка: gradF = (%.5f; %.5f),  -A^T u = (%.5f; %.5f)" % (l[0], l[1], r[0], r[1]))
print("Нарушение: g1 = %.2e" % max(0.0, A1[0] @ x1 - b1[0]))

print("\n--- второй старт внутри Omega: (12; 0), стена 1 ---")
x1b, u1b, I1b, path1b, n1b, uw1b = rosen(A1, b1, x0in, [1])
nf1b = NF
ng1b = NG
print("Ответ: x* = (%.6f; %.6f), F = %.6f, u = %s, шагов %d, NF = %d, NgradF = %d"
      % (x1b[0], x1b[1], F(x1b), str(np.round(uw1b, 6)), n1b, nf1b, ng1b))
print("Сравнение числа шагов: из x0 - %d, из (12; 0) - %d" % (n1, n1b))

# прогон 2
print("\n" + "=" * 78)
print("ПРОГОН 2: стены 1 + 2, старт x0 = (-4; -4)")
print("=" * 78)
A2 = A[[0, 1]]
b2 = b[[0, 1]]
x2, u2, I2, path2, n2, uw2 = rosen(A2, b2, x0, [1, 2])
nf2 = NF
ng2 = NG
print("Ответ: x* = (%.6f; %.6f), F = %.6f, u = %s, шагов %d, NF = %d, NgradF = %d"
      % (x2[0], x2[1], F(x2), str(np.round(uw2, 6)), n2, nf2, ng2))
l, r = check_kt(A2, I2, x2, u2)
print("Проверка: gradF = (%.5f; %.5f),  -A^T u = (%.5f; %.5f)" % (l[0], l[1], r[0], r[1]))
print("(Проекция старта на Omega сразу попала в угол, поэтому паркан не нужен.)")

print("\n--- стены 1 + 2 из внутреннего старта (12; 0): здесь работает паркан ---")
x2b, u2b, I2b, path2b, n2b, uw2b = rosen(A2, b2, x0in, [1, 2])
nf2b = NF
ng2b = NG
print("Ответ: x* = (%.6f; %.6f), F = %.6f, u = %s, шагов %d, NF = %d, NgradF = %d"
      % (x2b[0], x2b[1], F(x2b), str(np.round(uw2b, 6)), n2b, nf2b, ng2b))
l, r = check_kt(A2, I2b, x2b, u2b)
print("Проверка: gradF = (%.5f; %.5f),  -A^T u = (%.5f; %.5f)" % (l[0], l[1], r[0], r[1]))
print("Все gj <= 1e-9:", feasible(A2, b2, x2b))

# прогон 3
print("\n" + "=" * 78)
print("ПРОГОН 3: стены 1 + 3, старт в углу (6; 0) -> отпускание стены")
print("=" * 78)
A3 = A[[0, 2]]
b3 = b[[0, 2]]
x3, u3, I3, path3, n3, uw3 = rosen(A3, b3, xcorner, [1, 3])
nf3 = NF
ng3 = NG
print("Ответ: x* = (%.6f; %.6f), F = %.6f, u = %s, шагов %d, NF = %d, NgradF = %d"
      % (x3[0], x3[1], F(x3), str(np.round(uw3, 6)), n3, nf3, ng3))
print("Ответ совпадает с прогоном 1: x* = (3; 1.5); множители (стена 1; стена 3) = (%.4f; %.4f) - стена 3 не активна" % (uw3[0], uw3[1]))

# сравнение с ЛР 11
print("\n" + "=" * 78)
print("СРАВНЕНИЕ С ЛР 11 (штрафы: c = 10, eps = 0.01, внутри Флетчер-Ривс)")
print("=" * 78)
print("Стена 1:")
print("  %-26s %6s %6s %6s  %-22s %10s %10s" % ("метод", "итер", "NF", "NgF", "ответ", "нарушение", "lambda"))
print("  %-26s %6d %6d %6d  (%.5f; %.5f)  %10.1e %10.4f"
      % ("штраф ЛР 11 (внеш.)", 3, 241, 11, 2.99834, 1.49834, 4.99e-3, 0.9977))
print("  %-26s %6d %6d %6d  (%.5f; %.5f)  %10.1e %10.4f"
      % ("Розен, старт x0", n1, nf1, ng1, x1[0], x1[1], max(0.0, A1[0] @ x1 - b1[0]), uw1[0]))
print("  %-26s %6d %6d %6d  (%.5f; %.5f)  %10.1e %10.4f"
      % ("Розен, старт (12; 0)", n1b, nf1b, ng1b, x1b[0], x1b[1], max(0.0, A1[0] @ x1b - b1[0]), uw1b[0]))
print("Стены 1 + 2:")
print("  %-26s %6d %6d %6d  (%.5f; %.5f)  %10.1e  (%.3f; %.3f)"
      % ("штраф ЛР 11 (внеш.)", 5, 416, 20, 3.99820, 1.00075, 7.49e-4, 5.989, 14.976))
print("  %-26s %6d %6d %6d  (%.5f; %.5f)  %10.1e  (%.3f; %.3f)"
      % ("Розен, старт x0", n2, nf2, ng2, x2[0], x2[1], 0.0, uw2[0], uw2[1]))
print("  %-26s %6d %6d %6d  (%.5f; %.5f)  %10.1e  (%.3f; %.3f)"
      % ("Розен, старт (12; 0)", n2b, nf2b, ng2b, x2b[0], x2b[1], 0.0, uw2b[0], uw2b[1]))

# круг
print("\n" + "=" * 78)
print("КРУГ x1^2 + x2^2 >= 16: проекция точки, h = 0.1, eps = 0.01")
print("=" * 78)
R = 4.0
print("Точные минимумы на окружности: A = (3.6513; 1.6333), F = -8.1039; B = (-2.5119; -3.1130), F = 49.2597")
circ_starts = [np.array([-4.0, -4.0]), np.array([5.0, 3.0])]
circ_paths = []
for s in circ_starts:
    xc, kc, pc = circle_projection(s, R)
    circ_paths.append(pc)
    print("старт (%.0f; %.0f): x = (%.5f; %.5f), |x| = %.5f, F = %.5f, шагов %d, NgradF = %d"
          % (s[0], s[1], xc[0], xc[1], np.linalg.norm(xc), F(xc), kc, NG))


# ГРАФИКИ
def base(ax, xlim, ylim, levels):
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

def draw(ax, path, style, label):
    p = np.array(path)
    ax.plot(p[:, 0], p[:, 1], style, ms=6, label=label)

# График 1
fig1, ax = plt.subplots(figsize=(8, 7))
X1, X2 = base(ax, (-6, 14), (-6, 8), [-9.5, -9, -5, 0, 20, 60, 120, 200])
ax.contourf(X1, X2, 6 - X1 - 2*X2, levels=[-1000, 0], colors=["#c8f0c8"], alpha=0.5)
t = np.linspace(-6, 14, 10)
ax.plot(t, (6 - t)/2, "g-", lw=2, label="стена 1: x1 + 2x2 = 6")
ax.plot(*x0, "ks", label="x0 = (-4; -4)")
draw(ax, [x0] + path1, "r.-", "Розен из x0 (шаг 0 + спуск по стене)")
draw(ax, path1b, "m.-", "Розен из (12; 0)")
lr11 = np.array([[2.875006, 1.375017], [2.983855, 1.483839], [2.998338, 1.498337]])
ax.plot(lr11[:, 0], lr11[:, 1], "o", color="orange", mfc="none", ms=9, label="x(r) из ЛР 11 (r = 1, 10, 100)")
ax.plot(2.5, 1.0, "kx", ms=9, label="безусловный минимум (2.5; 1)")
ax.plot(3, 1.5, "g*", ms=14, label="x* = (3; 1.5)")
ax.set_title("Метод Розена со стеной 1 против штрафа ЛР 11")
ax.legend(loc="upper right", fontsize=7)
fig1.savefig("lr12_wall1.png", dpi=130)

# График 2
fig2, ax = plt.subplots(figsize=(8, 7))
X1, X2 = base(ax, (-6, 14), (-6, 8), [-9, -6, 0, 20, 60, 120, 200])
ok = np.maximum(6 - X1 - 2*X2, X2 - 1)
ax.contourf(X1, X2, ok, levels=[-1000, 0], colors=["#c8f0c8"], alpha=0.5)
ax.plot(t, (6 - t)/2, "g-", lw=2, label="стена 1")
ax.axhline(1, color="orange", lw=2, label="стена 2: x2 = 1")
ax.plot(*x0, "ks", label="x0 = (-4; -4)")
draw(ax, [x0] + path2, "r.-", "Розен из x0 (проекция сразу в угол)")
draw(ax, path2b, "m.-", "Розен из (12; 0): паркан у стены 2")
lr11b = np.array([[2.89130, 1.32609], [3.31819, 1.27273], [3.84558, 1.06407], [3.98230, 1.00737], [3.99820, 1.00075]])
ax.plot(lr11b[:, 0], lr11b[:, 1], "o-", color="orange", mfc="none", ms=8, lw=0.8, label="x(r) из ЛР 11")
ax.plot(3, 1.5, "b^", label="только стена 1: (3; 1.5)")
ax.plot(4, 1, "g*", ms=14, label="x* = (4; 1), u = (6; 15)")
ax.set_title("Две стены: паркан и множители в углу")
ax.legend(loc="upper right", fontsize=7)
fig2.savefig("lr12_walls12.png", dpi=130)

# График 3
fig3, ax = plt.subplots(figsize=(7, 6))
X1, X2 = base(ax, (0, 9), (-2, 5), [-9.5, -9, -8, -5, 0, 20, 60])
ok = np.maximum(6 - X1 - 2*X2, -X2)
ax.contourf(X1, X2, ok, levels=[-1000, 0], colors=["#c8f0c8"], alpha=0.5)
t3 = np.linspace(0, 9, 10)
ax.plot(t3, (6 - t3)/2, "g-", lw=2, label="стена 1")
ax.axhline(0, color="gray", lw=2, label="стена 3: x2 = 0")
draw(ax, path3, "r.-", "Розен: u3 < 0 -> стена 3 отпущена")
ax.plot(*xcorner, "ks", label="старт в углу (6; 0)")
ax.plot(3, 1.5, "g*", ms=14, label="x* = (3; 1.5)")
ax.set_title("Отпускание стены: прогон 3")
ax.legend(loc="upper right", fontsize=8)
fig3.savefig("lr12_release.png", dpi=130)

# График 4
fig4, ax = plt.subplots(figsize=(7.5, 7))
base(ax, (-7, 7), (-7, 7), [-8, 0, 20, 49.26, 80, 150])
a = np.linspace(0, 2*np.pi, 300)
ax.plot(R*np.cos(a), R*np.sin(a), "g-", lw=2, label="окружность |x| = 4 (Omega снаружи)")
cols = ["r", "m"]
for i in range(2):
    draw(ax, circ_paths[i], ".-", "старт (%.0f; %.0f)" % (circ_starts[i][0], circ_starts[i][1]))
    ax.plot(*circ_starts[i], "s", color="k")
ax.plot(3.6513, 1.6333, "g*", ms=14, label="A: F = -8.104")
ax.plot(-2.5119, -3.1130, "y*", ms=14, mec="k", label="B: F = 49.26")
ax.set_title("Круг: проекция точки, h = 0.1")
ax.legend(loc="upper left", fontsize=8)
fig4.savefig("lr12_circle.png", dpi=130)

print("\nГрафики сохранены: lr12_wall1.png, lr12_walls12.png, lr12_release.png, lr12_circle.png")
plt.show()
