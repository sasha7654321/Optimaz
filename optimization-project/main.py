"""Проєктна робота «Застосування методів оптимізації до відновлення цифрового зображення».
Варіант №4. Запуск:  python main.py        (повний прогін, ~2-3 хв)
                     python main.py --fast (швидкий прогін, грубіші сітки для графіків)
Усі результати (картинки, графіки, таблиці) потрапляють у теку outputs/.
"""
import csv
import math
import os
import sys
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.variant import (SEED, NOISE_SIGMA, X0, DELTA0, EPS, LAMBDA0, SIMPLEX_EDGE, SIZE,
                         STUDENT_ID)
from src.image import clean_image, add_noise, save_png
from src.metrics import mse, psnr
from src.oracle import Oracle
from src.filters import gaussian_blur, gaussian_kernel, bilateral_filter
from src.lab1_svenn import svenn
from src.lab2_search import dichotomy, golden_section, TAU
from src.lab3_nelder_mead import nelder_mead, LO as LO2, HI as HI2
from src.lab3_powell import powell, line_minimize
from src.operators import (energy, energy_grad, tikhonov_exact, check_adjoint, check_gradient)
from src.lab4_gradient import gradient_method, tv_denoise
from src.lab5_constrained import (project, violation, projected_gradient, penalty_method)

FAST = "--fast" in sys.argv
OUT = "outputs"
os.makedirs(OUT, exist_ok=True)
plt.rcParams["font.family"] = "DejaVu Sans"

MD = []          # зведений markdown зі всіма таблицями
CHECKS = []      # звірка з контрольними числами варіанта


def md(text=""):
    MD.append(text)


def table(headers, rows):
    headers = [str(h) for h in headers]
    out = ["| " + " | ".join(headers) + " |", "|" + "|".join(["---"] * len(headers)) + "|"]
    for r in rows:
        out.append("| " + " | ".join(str(c) for c in r) + " |")
    return "\n".join(out)


def write_csv(name, headers, rows):
    with open(os.path.join(OUT, name), "w", newline="", encoding="utf-8-sig") as fh:
        w = csv.writer(fh, delimiter=";")
        w.writerow(headers)
        w.writerows(rows)


def check(name, got, expected, tol=None):
    """Додає рядок у звірку з контрольними числами."""
    if isinstance(expected, (int, str)) and not isinstance(expected, bool):
        ok = (got == expected)
    else:
        ok = abs(got - expected) <= (tol if tol is not None else 6e-5 * max(1.0, abs(expected)))
    CHECKS.append((name, got, expected, ok))
    return ok


def fmt(x, d=6):
    return f"{x:.{d}f}"


def section(title):
    print(f"\n=== {title} ===", flush=True)
    md(f"\n## {title}\n")


t_start = time.time()

# ──────────────────────────────────────────────────────────────────────────────
# Частина 0
# ──────────────────────────────────────────────────────────────────────────────
section("Частина 0. Підготовка")
clean = clean_image(SIZE)
raw, noisy = add_noise(clean, NOISE_SIGMA, SEED)
save_png(clean, f"{OUT}/00_clean.png")
save_png(noisy, f"{OUT}/00_noisy.png")

mse0, psnr0 = mse(noisy, clean), psnr(noisy, clean)
n_out_raw = int(((raw < 0) | (raw > 1)).sum())
print(f"seed={SEED}  MSE={mse0:.8f}  PSNR={psnr0:.4f}  raw поза [0,1]: {n_out_raw}")
md(table(["Що", "Значення"], [
    ["зерно", SEED], ["σ шуму", NOISE_SIGMA],
    ["MSE зіпсованого", f"{mse0:.8f}"], ["PSNR зіпсованого", f"{psnr0:.4f} дБ"],
    ["raw поза [0,1]", f"{n_out_raw} ({100 * n_out_raw / SIZE ** 2:.2f} %)"],
    ["мін / макс raw", f"{raw.min():.4f} / {raw.max():.4f}"]]))
check("зерно", SEED, 136405)
check("MSE зіпсованого", mse0, 0.01263792, 5e-9)
check("PSNR зіпсованого", psnr0, 18.9832, 5e-5)
check("raw поза [0,1]", n_out_raw, 1571)

# ──────────────────────────────────────────────────────────────────────────────
# Лабораторна 1 — Свенн
# ──────────────────────────────────────────────────────────────────────────────
section("Лабораторна 1. Алгоритм Свенна")
LO1, HI1 = 0.05, 8.0


def f1(sigma):
    c = min(max(sigma, LO1), HI1)
    return mse(gaussian_blur(noisy, c), clean) + 1000.0 * (sigma - c) ** 2


oracle1 = Oracle(f1)
sv = svenn(oracle1, X0, DELTA0, LO1, HI1)
rows1 = [[r["k"], fmt(r["x"], 4), f"{r['f']:.8f}", fmt(r["delta"], 4), r["p"], r["comment"]]
         for r in sv["rows"]]
md(table(["k", "x_k", "f(x_k)", "Δ_k", "p", "Коментар"], rows1))
md(f"\nвідрізок = [{sv['a']:.4f}; {sv['b']:.4f}], довжина = {sv['b'] - sv['a']:.4f}, "
   f"N_f = {sv['n_f']}, p = {sv['p']}")
write_csv("lab1_svenn_table.csv", ["k", "x", "f", "delta", "p", "comment"],
          [[r["k"], r["x"], r["f"], r["delta"], r["p"], r["comment"]] for r in sv["rows"]])
print(f"Свенн: [{sv['a']:.4f}; {sv['b']:.4f}]  N_f={sv['n_f']}")
check("Свенн: a0", sv["a"], 0.3215, 5e-5)
check("Свенн: b0", sv["b"], 0.8317, 5e-5)
check("Свенн: N_f", sv["n_f"], 3)

k_demo = gaussian_kernel(X0)
md(f"\nЯдро для σ = x0 = {X0}: {np.round(k_demo, 6).tolist()}")
md(f"Перевірка: рівне поле 0.5 після розмиття, макс. відхилення = "
   f"{np.abs(gaussian_blur(np.full((SIZE, SIZE), 0.5), 0.7) - 0.5).max():.1e}")

grid1 = np.linspace(0.05, 8.0, 200)
vals1 = [oracle1.peek(s) for s in grid1]                 # peek — не рахується!
plt.figure(figsize=(9, 5))
plt.plot(grid1, vals1, label="f(σ)")
plt.scatter([r["x"] for r in sv["rows"]], [r["f"] for r in sv["rows"]], color="red", zorder=5,
            label="пробні точки Свенна")
plt.axvspan(sv["a"], sv["b"], alpha=0.2, color="green",
            label=f"знайдений відрізок [{sv['a']:.3f}; {sv['b']:.3f}]")
plt.xscale("log")
plt.xlabel("σ — сила розмиття"); plt.ylabel("f(σ) = MSE")
plt.title("Лабораторна 1: локалізація мінімуму методом Свенна")
plt.legend(); plt.grid(alpha=0.3)
plt.savefig(f"{OUT}/lab1_svenn.png", dpi=150, bbox_inches="tight"); plt.close()

# Обов'язкові досліди
md("\n**Обов'язкові досліди (лаб. 1)**\n")
exp_rows = []
for label, x0_, d_ in [("базовий запуск", X0, DELTA0), ("дослід 1: Δ0/10", X0, DELTA0 / 10),
                       ("дослід 2: Δ0·10", X0, DELTA0 * 10), ("дослід 3: x0 = 3.0", 3.0, DELTA0)]:
    o = Oracle(f1)
    r = svenn(o, x0_, d_, LO1, HI1)
    exp_rows.append([label, x0_, fmt(d_, 5), f"[{r['a']:.4f}; {r['b']:.4f}]",
                     fmt(r["b"] - r["a"], 4), r["p"], r["n_f"], r["status"]])
md(table(["Запуск", "x0", "Δ0", "відрізок", "довжина", "p", "N_f", "статус"], exp_rows))

# ──────────────────────────────────────────────────────────────────────────────
# Лабораторна 2 — дихотомія і золотий переріз
# ──────────────────────────────────────────────────────────────────────────────
section("Лабораторна 2. Дихотомія та золотий переріз")
a0, b0 = sv["a"], sv["b"]
oracle1.reset(); res_d = dichotomy(oracle1, a0, b0, EPS)
oracle1.reset(); res_g = golden_section(oracle1, a0, b0, EPS)
for name, r in (("Дихотомія", res_d), ("Золотий переріз", res_g)):
    print(f"{name:16s} σ*={r['x']:.6f} ітерацій={r['iterations']:2d} N_f={r['n_f']}")

L0 = b0 - a0


def compress(r):
    """Стиснення на одне звернення: (L_кінцева / L0)^(1/N_f)."""
    return ((r["b"] - r["a"]) / L0) ** (1.0 / r["n_f"])


md(table(["Метод", "σ*", "f(σ*)", "Довжина в кінці", "Ітерацій", "N_f", "Стиснення на 1 звернення"],
         [["Дихотомія", f"{res_d['x']:.6f}", f"{res_d['f']:.8f}", f"{res_d['b'] - res_d['a']:.3e}",
           res_d["iterations"], res_d["n_f"], f"{compress(res_d):.4f}"],
          ["Золотий переріз", f"{res_g['x']:.6f}", f"{res_g['f']:.8f}", f"{res_g['b'] - res_g['a']:.3e}",
           res_g["iterations"], res_g["n_f"], f"{compress(res_g):.4f}"]]))
check("Дихотомія: σ*", res_d["x"], 0.614215, 1.5e-6)
check("Дихотомія: ітерацій", res_d["iterations"], 17)
check("Дихотомія: N_f", res_d["n_f"], 35)
check("Золотий переріз: σ*", res_g["x"], 0.614218, 1.5e-6)
check("Золотий переріз: ітерацій", res_g["iterations"], 23)
check("Золотий переріз: N_f", res_g["n_f"], 26)

write_csv("lab2_dichotomy_table.csv", ["k", "a", "b", "x1", "x2", "f1", "f2", "L", "decision"],
          [[r["k"], r["a"], r["b"], r["x1"], r["x2"], r["f1"], r["f2"], r["L"], r["decision"]]
           for r in res_d["rows"]])
write_csv("lab2_golden_table.csv", ["k", "a", "b", "x1", "x2", "f1", "f2", "L", "new", "decision"],
          [[r["k"], r["a"], r["b"], r["x1"], r["x2"], r["f1"], r["f2"], r["L"], r["new"], r["decision"]]
           for r in res_g["rows"]])

md("\nПерші рядки таблиці дихотомії:\n")
md(table(["k", "a_k", "b_k", "x1", "x2", "f(x1)", "f(x2)", "L_k"],
         [[r["k"], f"{r['a']:.6f}", f"{r['b']:.6f}", f"{r['x1']:.6f}", f"{r['x2']:.6f}",
           f"{r['f1']:.8f}", f"{r['f2']:.8f}", f"{r['L']:.3e}"] for r in res_d["rows"][:6]]))
md("\nПерші рядки таблиці золотого перерізу:\n")
md(table(["k", "a_k", "b_k", "x1", "x2", "f(x1)", "f(x2)", "L_k", "нова точка"],
         [[r["k"], f"{r['a']:.6f}", f"{r['b']:.6f}", f"{r['x1']:.6f}", f"{r['x2']:.6f}",
           f"{r['f1']:.8f}", f"{r['f2']:.8f}", f"{r['L']:.3e}", f"{r['new']:.6f}"]
          for r in res_g["rows"][:6]]))
md("\nПовні таблиці — у файлах lab2_dichotomy_table.csv та lab2_golden_table.csv.")

sigma_d, sigma_g = res_d["x"], res_g["x"]
restored2 = gaussian_blur(noisy, sigma_g)
psnr2 = psnr(restored2, clean)
save_png(restored2, f"{OUT}/lab2_restored.png")
md(f"\nσ* = {sigma_g:.6f}: MSE {mse0:.8f} → {mse(restored2, clean):.8f}, "
   f"PSNR {psnr0:.4f} → {psnr2:.4f} дБ")
check("PSNR після лаб. 2", psnr2, 22.6893, 5e-5)

# Графік: довжина відрізка проти кількості звернень (ціна методу)
plt.figure(figsize=(9, 5))
plt.semilogy([0] + [2 * r["k"] for r in res_d["rows"]], [L0] + [r["L"] for r in res_d["rows"]],
             "o-", ms=3, label="дихотомія")
plt.semilogy([0] + [2 + r["k"] for r in res_g["rows"]], [L0] + [r["L"] for r in res_g["rows"]],
             "s-", ms=3, label="золотий переріз")
plt.axhline(EPS, color="gray", ls="--", label="ε")
plt.xlabel("кількість звернень до функції N_f"); plt.ylabel("довжина відрізка")
plt.title("Лабораторна 2: звуження відрізка залежно від кількості звернень")
plt.legend(); plt.grid(alpha=0.3)
plt.savefig(f"{OUT}/lab2_cost.png", dpi=150, bbox_inches="tight"); plt.close()

# Обов'язкові досліди
md("\n**Обов'язкові досліди (лаб. 2)**\n")
e1 = []
for eps_ in (1e-5, 1e-6):
    oracle1.reset(); rd = dichotomy(oracle1, a0, b0, eps_)
    oracle1.reset(); rg = golden_section(oracle1, a0, b0, eps_)
    e1.append([f"ε = {eps_:g}", rd["iterations"], rd["n_f"], rg["iterations"], rg["n_f"]])
md("Дослід 1. Точність ε:\n")
md(table(["", "дихотомія: ітер.", "дихотомія: N_f", "золотий: ітер.", "золотий: N_f"], e1))
oracle1.reset(); rdd = dichotomy(oracle1, a0, b0, EPS, delta=EPS, max_iter=500)
md(f"\nДослід 2. δ = ε у дихотомії: довжина після {rdd['iterations']} ітерацій = "
   f"{rdd['b'] - rdd['a']:.3e} (потрібно ≤ {EPS:g}), N_f = {rdd['n_f']} — метод "
   f"{'не зупинився сам (спрацював ліміт ітерацій)' if rdd['iterations'] >= 500 else 'зупинився'}.")
e3 = []
for s_ in (0.3, 0.5, 0.8, 1.5, sigma_g):
    e3.append([f"{s_:.4f}", f"{psnr(gaussian_blur(noisy, s_), clean):.4f}"])
md("\nДослід 3. Ручний підбір σ (останній рядок — оптимум):\n")
md(table(["σ", "PSNR, дБ"], e3))

# ──────────────────────────────────────────────────────────────────────────────
# Лабораторна 3 — Нелдер—Мід і Пауелл
# ──────────────────────────────────────────────────────────────────────────────
section("Лабораторна 3. Нелдер—Мід і Пауелл (білатеральний фільтр)")


def make_obj2(noisy_img, clean_img):
    def f(theta):
        p = np.asarray(theta, dtype=float)
        c = np.clip(p, LO2, HI2)
        value = mse(bilateral_filter(noisy_img, c[0], c[1]), clean_img)
        gap = p - c
        return value + 1000.0 * float(gap @ gap)
    return f


oracle2 = Oracle(make_obj2(noisy, clean))
# θ0 береться від результату лаб. 2 (σ* із таблиці, округлене до 6 знаків, як у практикумі).
# Функція тут кускова (радіус вікна цілий), тому N_f Нелдера—Міда чутливий до 7-го знака θ0.
theta0 = np.array([min(max(2 * round(sigma_d, 6), 0.5), 4.0), min(max(0.15 + 0.1 * NOISE_SIGMA, 0.02), 0.9)])
f_theta0 = oracle2.peek(theta0)
print(f"θ0 = ({theta0[0]:.6f}; {theta0[1]:.6f})  f(θ0)={f_theta0:.8f}")
md(f"θ0 = ({theta0[0]:.6f}; {theta0[1]:.6f}), f(θ0) = {f_theta0:.8f} "
   f"(найкраще гаусове розмиття давало {res_g['f']:.8f})")
check("θ0: σs", theta0[0], 1.2284, 5e-5)
check("θ0: σr", theta0[1], 0.1620, 5e-5)

oracle2.reset(); res_nm = nelder_mead(oracle2, theta0, SIMPLEX_EDGE)
oracle2.reset(); res_pw = powell(oracle2, theta0, collect_stats=True)
print(f"Нелдер—Мід: θ*=({res_nm['x'][0]:.4f}; {res_nm['x'][1]:.4f}) ітер={res_nm['iterations']} N_f={res_nm['n_f']}")
print(f"Пауелл:     θ*=({res_pw['x'][0]:.4f}; {res_pw['x'][1]:.4f}) циклів={res_pw['cycles']} N_f={res_pw['n_f']}")

md("\nНелдер—Мід, перші ітерації:\n")
md(table(["k", "операція", "найкраща вершина", "f", "розмір", "N_f"],
         [[r["k"], r["op"], f"({r['x'][0]:.4f}; {r['x'][1]:.4f})", f"{r['f']:.8f}", f"{r['size']:.5f}",
           r["n_f"]] for r in res_nm["rows"][:6] + res_nm["rows"][-1:]]))
md("\nПауелл, цикли:\n")
md(table(["цикл", "x", "f", "‖зсув‖", "α1", "α2", "N_f"],
         [[r["cycle"], f"({r['x'][0]:.4f}; {r['x'][1]:.4f})", f"{r['f']:.8f}", f"{r['shift']:.6f}",
           f"{r['alpha1']:+.5f}", f"{r['alpha2']:+.5f}", r["n_f"]] for r in res_pw["rows"]]))
md("\n" + table(["Метод", "σs*", "σr*", "f(θ*)", "Ітерацій (циклів)", "N_f"],
                [["Нелдер—Мід", f"{res_nm['x'][0]:.6f}", f"{res_nm['x'][1]:.6f}", f"{res_nm['f']:.10f}",
                  res_nm["iterations"], res_nm["n_f"]],
                 ["Пауелл", f"{res_pw['x'][0]:.6f}", f"{res_pw['x'][1]:.6f}", f"{res_pw['f']:.10f}",
                  res_pw["cycles"], res_pw["n_f"]]]))
write_csv("lab3_nelder_mead_table.csv", ["k", "op", "sigma_s", "sigma_r", "f", "size", "n_f"],
          [[r["k"], r["op"], r["x"][0], r["x"][1], r["f"], r["size"], r["n_f"]] for r in res_nm["rows"]])
write_csv("lab3_powell_table.csv", ["cycle", "sigma_s", "sigma_r", "f", "shift", "alpha1", "alpha2", "n_f"],
          [[r["cycle"], r["x"][0], r["x"][1], r["f"], r["shift"], r["alpha1"], r["alpha2"], r["n_f"]]
           for r in res_pw["rows"]])
check("Нелдер—Мід: σs*", res_nm["x"][0], 2.0649, 5e-5)
check("Нелдер—Мід: σr*", res_nm["x"][1], 0.2364, 5e-5)
check("Нелдер—Мід: ітерацій", res_nm["iterations"], 58)
check("Нелдер—Мід: N_f", res_nm["n_f"], 115)
check("Пауелл: циклів", res_pw["cycles"], 6)
check("Пауелл: N_f", res_pw["n_f"], 415)

best3 = res_nm["x"] if res_nm["f"] <= res_pw["f"] else res_pw["x"]
restored3 = bilateral_filter(noisy, best3[0], best3[1])
psnr3 = psnr(restored3, clean)
save_png(restored3, f"{OUT}/lab3_restored.png")
md(f"\nПісля відновлення: MSE = {mse(restored3, clean):.8f}, PSNR = {psnr3:.4f} дБ "
   f"(лаб. 2: {psnr2:.4f}, приріст {psnr3 - psnr2:+.2f} дБ)")
check("PSNR після лаб. 3", psnr3, 26.6577, 5e-5)

# Лінії рівня + траєкторії
n_grid = 14 if FAST else 40
ss = np.linspace(0.3, 5.0, n_grid)
sr = np.linspace(0.01, 1.0, n_grid)
grid3 = np.array([[oracle2.peek([a, b]) for a in ss] for b in sr])
plt.figure(figsize=(9, 6))
levels = np.unique(np.quantile(grid3, np.linspace(0.0, 0.6, 25)))
cs = plt.contourf(ss, sr, grid3, levels=levels, alpha=0.75)
plt.colorbar(cs, label="f(σ_s, σ_r) = MSE")
nm_path = np.array([theta0] + [r["x"] for r in res_nm["rows"]])
pw_path = np.array([theta0] + [r["x"] for r in res_pw["rows"]])
plt.plot(nm_path[:, 0], nm_path[:, 1], "o-", ms=3, lw=1, label="Нелдер—Мід")
plt.plot(pw_path[:, 0], pw_path[:, 1], "s-", ms=5, lw=1.5, label="Пауелл")
plt.scatter([theta0[0]], [theta0[1]], marker="^", s=90, color="k", zorder=6, label="θ0")
plt.scatter([res_nm["x"][0]], [res_nm["x"][1]], marker="*", s=220, color="gold", edgecolor="k",
            zorder=5, label="оптимум")
plt.xlabel("σ_s — просторовий масштаб"); plt.ylabel("σ_r — фотометричний поріг")
plt.title("Лабораторна 3: лінії рівня і траєкторії методів")
plt.legend(); plt.savefig(f"{OUT}/lab3_contours.png", dpi=150, bbox_inches="tight"); plt.close()

# Обов'язкові досліди
md("\n**Обов'язкові досліди (лаб. 3)**\n")
md("Дослід 1. Розмір початкового трикутника:\n")
e31 = []
for label, s_ in [("s/3", SIMPLEX_EDGE / 3), ("s (з варіанта)", SIMPLEX_EDGE), ("3s", SIMPLEX_EDGE * 3)]:
    oracle2.reset(); r = nelder_mead(oracle2, theta0, s_)
    e31.append([label, f"{s_:.4f}", f"({r['x'][0]:.4f}; {r['x'][1]:.4f})", f"{r['f']:.10f}", r["iterations"], r["n_f"]])
md(table(["Ребро", "значення", "θ*", "f(θ*)", "ітерацій", "N_f"], e31))

md("\nДослід 2. З чого складається вартість першого циклу Пауелла (звернень до f):\n")
st = res_pw["line_stats"][:3]
e32 = [[f"лінійний пошук {i + 1}", s["svenn"], s["golden"], s["check"], s["svenn"] + s["golden"] + s["check"]]
       for i, s in enumerate(st)]
tot = sum(s["svenn"] + s["golden"] + s["check"] for s in st)
e32.append(["разом (з f(θ0), що рахується один раз)", "", "", "", tot + 1])
md(table(["", "Свенн", "золотий переріз (разом із кінцевою точкою)", "перевірка нової точки", "усього"], e32))
md("\n(Перші два рядки — це «ітерація» по осях; третій — крок уздовж нового напрямку. "
   "У таблиці циклів вище N_f після першого циклу враховує лише перші два пошуки.)")

oracle2.reset(); pw_tight = powell(oracle2, theta0, tol=1e-7)
md(f"\nДослід 3. Критерій зупинки Пауелла 1e-7 замість 1e-4: циклів {pw_tight['cycles']}, "
   f"N_f {pw_tight['n_f']}, f = {pw_tight['f']:.10f} (базовий запуск: {res_pw['cycles']} циклів, "
   f"N_f {res_pw['n_f']}, f = {res_pw['f']:.10f}).")

# Дослід 4: одна ручка проти двох (σr зафіксовано)
sr_fixed = float(res_nm["x"][1])
f_1d = Oracle(lambda s: oracle2.peek([s, sr_fixed]) if LO2[0] <= s <= HI2[0]
              else oracle2.peek([s, sr_fixed]))
br4 = svenn(f_1d, float(theta0[0]), 0.1, float(LO2[0]), float(HI2[0]))
gs4 = golden_section(f_1d, br4["a"], br4["b"], 1e-4)
md(f"\nДослід 4. σr = {sr_fixed:.4f} зафіксовано, шукаємо лише σs (Свенн + золотий переріз): "
   f"σs = {gs4['x']:.4f}, f = {gs4['f']:.10f}, N_f = {f_1d.n_f}. Двовимірний оптимум: "
   f"f = {res_nm['f']:.10f}. Різниця: {gs4['f'] - res_nm['f']:+.2e}.")

# ──────────────────────────────────────────────────────────────────────────────
# Лабораторна 4 — градієнтні методи
# ──────────────────────────────────────────────────────────────────────────────
section("Лабораторна 4. Градієнтні методи (енергія Тихонова)")
adj = check_adjoint()
gchk = check_gradient(noisy[:16, :16], 0.3)
print(f"перевірка спряженості: {adj:.1e};  перевірка градієнта: {gchk:.1e}")
md(f"Перевірка 1 (спряженість grad/div): нев'язка = {adj:.2e}\n\n"
   f"Перевірка 2 (градієнт проти скінченних різниць): відносна різниця = {gchk:.2e}")
check("перевірка спряженості < 1e-12", float(adj < 1e-12), 1.0)
check("перевірка градієнта < 1e-6", float(gchk < 1e-6), 1.0)

LAM_LO, LAM_HI = 1e-4, 50.0


def g_lambda(lam):
    c = min(max(lam, LAM_LO), LAM_HI)
    return mse(tikhonov_exact(noisy, c), clean) + 1000.0 * (lam - c) ** 2


or_lam = Oracle(g_lambda)
sv_l = svenn(or_lam, LAMBDA0, 0.2, LAM_LO, LAM_HI)
gs_l = golden_section(or_lam, sv_l["a"], sv_l["b"], 1e-3)
lam = gs_l["x"]
L = 1.0 + 8.0 * lam
mu = 1.0
kappa = L / mu
h_opt = 1.0 / L
beta_hb = ((math.sqrt(L) - math.sqrt(mu)) / (math.sqrt(L) + math.sqrt(mu))) ** 2
beta_nag = (math.sqrt(L) - math.sqrt(mu)) / (math.sqrt(L) + math.sqrt(mu))
print(f"λ* = {lam:.6f}  L = {L:.6f}  β_кулька = {beta_hb:.6f}  β_Нестеров = {beta_nag:.6f}")
md(f"\nПошук λ (Свенн + золотий переріз, x0 = λ0 = {LAMBDA0}, Δ = 0.2, ε = 1e-3): "
   f"відрізок [{sv_l['a']:.6f}; {sv_l['b']:.6f}] ({sv_l['n_f']} звернень), "
   f"λ* = {lam:.6f} ({gs_l['iterations']} ітерацій, разом {or_lam.n_f} звернень), g(λ*) = {gs_l['f']:.8f}\n")
md(table(["Величина", "Значення"], [
    ["L = 1 + 8λ*", f"{L:.6f}"], ["μ", "1.0"], ["κ = L/μ", f"{kappa:.6f}"],
    ["h = 1/L", f"{h_opt:.6f}"], ["межа стійкості 2/L", f"{2 / L:.6f}"],
    ["β (важка кулька)", f"{beta_hb:.6f}"], ["β (Нестеров, сильно опуклий)", f"{beta_nag:.6f}"]]))
check("λ*", lam, 0.357461, 5e-7)
check("L", L, 3.859685, 5e-6)

I_star = tikhonov_exact(noisy, lam)
E_star = energy(I_star, noisy, lam)
gnorm_star = float(np.linalg.norm(energy_grad(I_star, noisy, lam)))
md(f"\nЕталонний розв'язок (ДКП): E* = {E_star:.8f}, ‖∇E(I*)‖ = {gnorm_star:.1e}, "
   f"E(I0) = {energy(noisy, noisy, lam):.8f}")
check("E*", E_star, 101.115873, 5e-7)

methods = [
    ("Спуск, h=1/L", dict(kind="gd")),
    ("Спуск, Армійо", dict(kind="armijo")),
    (f"Важка кулька, β={beta_hb:.4f}", dict(kind="hb", beta=beta_hb)),
    ("Важка кулька, β=0,9", dict(kind="hb", beta=0.9)),
    ("Нестеров, опуклий", dict(kind="nag")),
    ("Нестеров, сильно опуклий", dict(kind="nag", beta=beta_nag)),
]
results4 = {}
for name, kw in methods:
    results4[name] = gradient_method(noisy, lam, L, x_ref=I_star, **kw)

md("\n" + table(["Метод", "Ітерацій", "N_f", "N_∇f", "Разом", "E(I_N)", "‖∇E‖", "Збігся"],
                [[n, r["iterations"] if r["converged"] else 200, r["n_f"], r["n_grad"],
                  r["n_f"] + r["n_grad"], f"{r['E']:.8f}", f"{r['grad']:.1e}",
                  "так" if r["converged"] else "ні"] for n, r in results4.items()]))
md("\n(Для методу, що не збігся, у стовпці «Ітерацій» вказано ліміт 200.)")
write_csv("lab4_summary.csv", ["method", "iterations", "n_f", "n_grad", "E", "grad_norm", "converged"],
          [[n, r["iterations"], r["n_f"], r["n_grad"], r["E"], r["grad"], r["converged"]]
           for n, r in results4.items()])
print("Лаб. 4:", {n: (r["iterations"], r["converged"]) for n, r in results4.items()})
check("Спуск h=1/L: ітерацій", results4["Спуск, h=1/L"]["iterations"], 43)
check("Кулька (опт. β): ітерацій", results4[f"Важка кулька, β={beta_hb:.4f}"]["iterations"], 36)
check("Кулька β=0,9: не збіглася", float(results4["Важка кулька, β=0,9"]["converged"]), 0.0)
check("Нестеров сильно опуклий: ітерацій", results4["Нестеров, сильно опуклий"]["iterations"], 21)
check("β оптимальне", beta_hb, 0.1059, 5e-5)

g0 = results4["Спуск, h=1/L"]
md("\nТаблиця градієнтного спуску (перші рядки):\n")
md(table(["k", "E(I_k)", "‖∇E(I_k)‖", "h", "‖I_k − I*‖", "E − E*"],
         [[r["k"], f"{r['E']:.8f}", f"{r['grad']:.4e}", f"{r['h']:.6f}", f"{r['dist']:.4e}",
           f"{r['E'] - E_star:.4e}"] for r in g0["rows"][:7] + g0["rows"][-1:]]))
write_csv("lab4_gd_table.csv", ["k", "E", "grad_norm", "h", "dist_to_Istar", "E_minus_Estar"],
          [[r["k"], r["E"], r["grad"], r["h"], r["dist"], r["E"] - E_star] for r in g0["rows"]])

for key, ylab, fname in (("E", "E(I_k) − E*", "lab4_convergence_f.png"),
                         ("dist", "‖I_k − I*‖", "lab4_convergence_dist.png")):
    plt.figure(figsize=(9, 5))
    for name, r in results4.items():
        ys = [max((row["E"] - E_star) if key == "E" else row["dist"], 1e-16) for row in r["rows"]]
        plt.semilogy(range(len(ys)), ys, label=name)
    plt.xlabel("номер ітерації k"); plt.ylabel(ylab)
    plt.title("Лабораторна 4: збіжність " + ("за значенням функції" if key == "E" else "за відстанню до I*"))
    plt.legend(); plt.grid(alpha=0.3)
    plt.savefig(f"{OUT}/{fname}", dpi=150, bbox_inches="tight"); plt.close()

best4 = results4["Нестеров, сильно опуклий"]["I"]
restored4 = np.clip(best4, 0.0, 1.0)
psnr4 = psnr(restored4, clean)
save_png(restored4, f"{OUT}/lab4_restored.png")
md(f"\nPSNR попіксельного відновлення (з обрізанням до [0,1]) = {psnr4:.4f} дБ "
   f"(лаб. 3: {psnr3:.4f}, лаб. 2: {psnr2:.4f}, зіпсоване: {psnr0:.4f})")
check("PSNR після лаб. 4", psnr4, 22.5139, 5e-5)

# Додаткове завдання: повна варіація
lam_tv = max(0.02, 0.15 * lam)
tv_iters = 300 if FAST else 1500
I_tv = np.clip(tv_denoise(noisy, lam_tv, 5e-3, n_iter=tv_iters), 0, 1)
psnr_tv = psnr(I_tv, clean)
save_png(I_tv, f"{OUT}/lab4_tv_restored.png")
md(f"\nДодаткове завдання (повна варіація): λ_TV = {lam_tv:.5f}, ε = 5e-3, {tv_iters} ітерацій Нестерова → "
   f"PSNR = {psnr_tv:.4f} дБ (квадратичний штраф: {psnr4:.4f}, приріст {psnr_tv - psnr4:+.2f} дБ)")

# Обов'язкові досліди
md("\n**Обов'язкові досліди (лаб. 4)**\n")
lam6, L6 = 6.0, 1 + 8 * 6.0
b_hb6 = ((math.sqrt(L6) - 1) / (math.sqrt(L6) + 1)) ** 2
b_nag6 = (math.sqrt(L6) - 1) / (math.sqrt(L6) + 1)
e41 = []
for name, kw in (("Градієнтний спуск", dict(kind="gd")),
                 ("Важка кулька, оптимальне β", dict(kind="hb", beta=b_hb6)),
                 ("Нестеров, сильно опуклий", dict(kind="nag", beta=b_nag6))):
    r = gradient_method(noisy, lam6, L6, max_iter=3000, **kw)
    e41.append([name, r["iterations"] if r["converged"] else "3000 (не збігся)"])
md(f"Дослід 1. λ = 6, κ = {L6:.0f}, ліміт 3000 ітерацій (порівняйте з κ = {kappa:.2f}):\n")
md(table(["Метод", "Ітерацій при κ=49"], e41))

r2 = gradient_method(noisy, lam, L, kind="gd", step=2.1 / L, max_iter=50, tol=0.0)
md("\nДослід 2. Спуск із h = 2.1/L, значення E(I_k):\n")
md(table(["k", 0, 10, 20, 30, 40], [["E"] + [f"{r2['rows'][k]['E']:.4g}" for k in (0, 10, 20, 30, 40)]]))

lam_grid = np.logspace(-2.5, 1.5, 15 if FAST else 45)
psnr_l = [psnr(np.clip(tikhonov_exact(noisy, l), 0, 1), clean) for l in lam_grid]
e43 = [[f"{l:.2f}", f"{psnr(np.clip(tikhonov_exact(noisy, l), 0, 1), clean):.4f}"]
       for l in (0.05, 0.24, 1.0, 5.0)] + [[f"{lam:.4f} (λ*)", f"{psnr4:.4f}"]]
md("\nДослід 3. Вплив λ на якість:\n")
md(table(["λ", "PSNR, дБ"], e43))
plt.figure(figsize=(9, 5))
plt.semilogx(lam_grid, psnr_l, "-", label="PSNR(λ)")
plt.axvline(lam, color="red", ls="--", label=f"λ* = {lam:.3f}")
plt.axhline(psnr0, color="gray", ls=":", label="зіпсоване зображення")
plt.xlabel("λ"); plt.ylabel("PSNR, дБ")
plt.title("Лабораторна 4: якість відновлення залежно від λ")
plt.legend(); plt.grid(alpha=0.3)
plt.savefig(f"{OUT}/lab4_psnr_vs_lambda.png", dpi=150, bbox_inches="tight"); plt.close()

# ──────────────────────────────────────────────────────────────────────────────
# Лабораторна 5 — умовна оптимізація
# ──────────────────────────────────────────────────────────────────────────────
section("Лабораторна 5. Умовна оптимізація")
I_unc = tikhonov_exact(raw, lam)
v_unc = violation(I_unc)
E_unc = energy(I_unc, raw, lam)
md(f"Сире спостереження: поза [0,1] — {n_out_raw} пікселів, мін = {raw.min():.4f}, макс = {raw.max():.4f}\n\n"
   f"Безумовний розв'язок: поза Ω — {v_unc['count']} пікселів, максимальне порушення = {v_unc['max']:.6f}, "
   f"E = {E_unc:.8f}")
check("безумовний: пікселів поза Ω", v_unc["count"], 570)
check("безумовний: макс. порушення", v_unc["max"], 0.161994, 5e-7)

plt.figure(figsize=(9, 4))
plt.hist(raw.ravel(), bins=120, alpha=0.55, label="сире спостереження")
plt.hist(I_unc.ravel(), bins=120, alpha=0.55, label="безумовний розв'язок")
plt.axvline(0.0, color="red", ls="--", label="межі допустимої області")
plt.axvline(1.0, color="red", ls="--")
plt.xlabel("яскравість пікселя"); plt.ylabel("кількість пікселів")
plt.title("Лабораторна 5: гістограма яскравостей")
plt.legend(); plt.savefig(f"{OUT}/lab5_histogram.png", dpi=150, bbox_inches="tight"); plt.close()

res_pg = projected_gradient(raw, lam, L)
res_fi = projected_gradient(raw, lam, L, accelerated=True)
res_pen = penalty_method(raw, lam, L)
n_tot = lambda r: r["n_f"] + r["n_grad"]
print(f"Проєкція: {res_pg['iterations']} ітер., разом {n_tot(res_pg)};  FISTA: {res_fi['iterations']} / {n_tot(res_fi)};"
      f"  штраф: {len(res_pen['rows'])} зовн., разом {n_tot(res_pen)}")

md("\nПроєкція градієнта:\n")
md(table(["k", "E(I_k)", "‖G_h‖", "активних обмежень", "порушення"],
         [[r["k"], f"{r['E']:.8f}", f"{r['G']:.4e}", r["active"], r["violation"]]
          for r in res_pg["rows"][:6] + res_pg["rows"][-1:]]))
md("\nШтрафний метод:\n")
md(table(["s", "r_s", "E", "P", "r·P", "порушення", "поза Ω"],
         [[r["s"], f"{r['r']:g}", f"{r['E']:.8f}", f"{r['P']:.3e}", f"{r['rP']:.3e}",
           f"{r['violation']:.2e}", r["outside"]] for r in res_pen["rows"]]))
write_csv("lab5_projection_table.csv", ["k", "E", "G", "grad", "active", "violation", "n_f", "n_grad"],
          [[r["k"], r["E"], r["G"], r["grad"], r["active"], r["violation"], r["n_f"], r["n_grad"]]
           for r in res_pg["rows"]])
write_csv("lab5_penalty_table.csv", ["s", "r", "L_r", "step", "E", "P", "rP", "violation", "outside",
                                      "n_f", "n_grad"],
          [[r["s"], r["r"], r["L_r"], r["step"], r["E"], r["P"], r["rP"], r["violation"], r["outside"],
            r["n_f"], r["n_grad"]] for r in res_pen["rows"]])

vp, vf, vn = violation(res_pg["I"]), violation(res_fi["I"]), violation(res_pen["I"])
md("\n" + table(["Метод", "E(I*)", "Макс. порушення", "Поза Ω", "Допустимий", "Ітерацій", "N_f", "N_∇f", "Разом"],
                [["Проєкція градієнта", f"{res_pg['E']:.8f}", f"{vp['max']:.0e}", vp["count"], "так",
                  res_pg["iterations"], res_pg["n_f"], res_pg["n_grad"], n_tot(res_pg)],
                 ["FISTA", f"{res_fi['E']:.8f}", f"{vf['max']:.0e}", vf["count"], "так",
                  res_fi["iterations"], res_fi["n_f"], res_fi["n_grad"], n_tot(res_fi)],
                 ["Квадратичний штраф", f"{res_pen['E']:.8f}", f"{vn['max']:.2e}", vn["count"], "ні",
                  f"{len(res_pen['rows'])} зовн.", res_pen["n_f"], res_pen["n_grad"], n_tot(res_pen)]]))
check("Проєкція: ітерацій", res_pg["iterations"], 45)
check("Проєкція: разом звернень", n_tot(res_pg), 92)
check("Штраф: зовнішніх ітерацій", len(res_pen["rows"]), 7)
check("Штраф: разом звернень", n_tot(res_pen), 694)
check("Проєкція: порушення = 0", vp["max"], 0.0, 0.0)

restored5 = res_pg["I"]
psnr5 = psnr(restored5, clean)
save_png(restored5, f"{OUT}/lab5_restored.png")
md(f"\nPSNR з обмеженнями = {psnr5:.4f} дБ (безумовний, лаб. 4: {psnr4:.4f})")
check("PSNR після лаб. 5", psnr5, 22.6141, 5e-5)

E5_star = res_pg["E"]
plt.figure(figsize=(9, 5))
for name, r in (("Проєкція градієнта", res_pg), ("FISTA", res_fi)):
    ys = [max(row["E"] - E5_star, 1e-16) for row in r["rows"]]
    plt.semilogy(range(len(ys)), ys, label=name)
plt.xlabel("номер ітерації"); plt.ylabel("E(I_k) − E*")
plt.title("Лабораторна 5: збіжність проєкційних методів")
plt.legend(); plt.grid(alpha=0.3)
plt.savefig(f"{OUT}/lab5_convergence.png", dpi=150, bbox_inches="tight"); plt.close()

# Карта активних обмежень
act = np.full((SIZE, SIZE), 0.5)
act[restored5 <= 1e-12] = 0.0
act[restored5 >= 1 - 1e-12] = 1.0
save_png(act, f"{OUT}/lab5_active_map.png")
md(f"\nАктивних обмежень у розв'язку: на нулі — {int((restored5 <= 1e-12).sum())}, "
   f"на одиниці — {int((restored5 >= 1 - 1e-12).sum())} пікселів (карта: lab5_active_map.png).")

# Обов'язкові досліди
md("\n**Обов'язкові досліди (лаб. 5)**\n")
bad = projected_gradient(raw, lam, L, stop="grad")
md(f"Дослід 1. Критерій зупинки: за ‖G_h‖ — {res_pg['iterations']} ітерацій (збігся); "
   f"за ‖∇E‖ — {bad['iterations']} ітерацій ({'збігся' if bad['converged'] else 'НЕ збігся, ліміт 200'}), "
   f"кінцева ‖∇E‖ = {bad['rows'][-1]['grad']:.3e}, ‖G_h‖ = {bad['G']:.3e}.")
e52 = []
for growth, outer in ((2.0, 21), (10.0, 7), (100.0, 4)):
    r = penalty_method(raw, lam, L, growth=growth, outer=outer)
    v = violation(r["I"])
    e52.append([f"{growth:g}", len(r["rows"]), f"{r['rows'][-1]['r']:.3g}", n_tot(r), f"{v['max']:.2e}", v["count"]])
md("\nДослід 2. Множник зростання штрафу (r зростає до ~1e6):\n")
md(table(["множник", "зовн. ітерацій", "кінцеве r", "разом звернень", "порушення", "поза Ω"], e52))
md("\nДослід 3. Ціна обумовленості штрафного методу:\n")
md(table(["s", "r", "L(Φ_r) = L + 2r", "крок 1/L(Φ_r)"],
         [[r["s"], f"{r['r']:g}", f"{r['L_r']:.6g}", f"{r['step']:.3e}"] for r in res_pen["rows"]]))

# ──────────────────────────────────────────────────────────────────────────────
# Підсумок
# ──────────────────────────────────────────────────────────────────────────────
section("Підсумок усього практикуму")
md(table(["Етап", "Метод", "Що підбирали", "Невідомих", "Звернень", "PSNR, дБ"], [
    ["Вхід", "—", "—", "—", "—", f"{psnr0:.4f}"],
    ["Лаб. 1", "Свенн", "відрізок для σ", 1, sv["n_f"], "—"],
    ["Лаб. 2", "золотий переріз", f"σ* = {sigma_g:.6f}", 1, res_g["n_f"], f"{psnr2:.4f}"],
    ["Лаб. 3", "Нелдер—Мід", "(σs, σr)", 2, res_nm["n_f"], f"**{psnr3:.4f}**"],
    ["Лаб. 4", "Нестеров", "усі пікселі", SIZE * SIZE,
     n_tot(results4["Нестеров, сильно опуклий"]), f"{psnr4:.4f}"],
    ["Лаб. 5", "проєкція градієнта", "усі пікселі, з обмеженнями", SIZE * SIZE, n_tot(res_pg), f"{psnr5:.4f}"]]))

# Зведена картинка
fig, axes = plt.subplots(2, 4, figsize=(14, 7.4))
panels = [("Чисте", clean), ("Зіпсоване", noisy, psnr0), ("Лаб. 2: гаусів фільтр", restored2, psnr2),
          ("Лаб. 3: білатеральний", restored3, psnr3), ("Лаб. 4: Тихонов", restored4, psnr4),
          ("Лаб. 4 (дод.): TV", I_tv, psnr_tv), ("Лаб. 5: з обмеженнями", restored5, psnr5),
          ("Карта активних обмежень", act)]
for ax, p in zip(axes.ravel(), panels):
    ax.imshow(p[1], cmap="gray", vmin=0, vmax=1)
    ax.set_title(p[0] + (f"\nPSNR = {p[2]:.2f} дБ" if len(p) > 2 else ""), fontsize=10)
    ax.axis("off")
plt.tight_layout()
plt.savefig(f"{OUT}/summary_images.png", dpi=150, bbox_inches="tight"); plt.close()

# Звірка з контрольними числами
lines = ["# Звірка з контрольними числами варіанта №4\n",
         "| Що | Моє значення | Контрольне | Збіг |", "|---|---|---|---|"]
for name, got, exp, ok in CHECKS:
    g_ = f"{got:.6g}" if isinstance(got, float) else str(got)
    e_ = f"{exp:.6g}" if isinstance(exp, float) else str(exp)
    lines.append(f"| {name} | {g_} | {e_} | {'✅' if ok else '❌'} |")
n_ok = sum(1 for c in CHECKS if c[3])
lines.append(f"\n**Збіглося {n_ok} з {len(CHECKS)}.**")
open(f"{OUT}/control_check.md", "w", encoding="utf-8").write("\n".join(lines))
open(f"{OUT}/results.md", "w", encoding="utf-8").write(
    f"# Результати проєкту, варіант №{STUDENT_ID}\n" + "\n".join(MD) + "\n")

print(f"\nЗвірка з контрольними числами: {n_ok}/{len(CHECKS)} збіглося")
for name, got, exp, ok in CHECKS:
    if not ok:
        print("  ❌", name, got, exp)
print(f"Готово за {time.time() - t_start:.0f} с. Результати — у теці {OUT}/")
