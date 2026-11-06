import numpy as np

from src.lab1_svenn import svenn
from src.lab2_search import golden_section
from src.oracle import Oracle

LO = np.array([0.3, 0.01])
HI = np.array([5.0, 1.0])


def line_minimize(oracle, x, direction, delta=0.1, eps=1e-4, alpha_max=1000.0, stats=None):
    """Мінімум уздовж прямої: Свенн (лаб.1) дає відрізок, золотий переріз (лаб.2) уточнює."""
    line = Oracle(lambda a: oracle.value(np.clip(x + a * direction, LO, HI)))

    bracket = svenn(line, 0.0, delta, -alpha_max, alpha_max)
    n_svenn = line.n_f
    result = golden_section(line, bracket["a"], bracket["b"], eps)
    n_golden = line.n_f - n_svenn

    alpha = result["x"]
    x_new = np.clip(x + alpha * direction, LO, HI)
    if stats is not None:
        stats.append({"svenn": n_svenn, "golden": n_golden, "check": 1})
    return x_new, oracle.value(x_new), alpha


def powell(oracle, x0, tol=1e-4, max_cycles=50, collect_stats=False):
    """Метод спряжених напрямків Пауелла."""
    x = np.asarray(x0, dtype=float)
    f_x = oracle.value(x)
    directions = [np.array([1.0, 0.0]), np.array([0.0, 1.0])]
    rows = []
    stats = [] if collect_stats else None

    for cycle in range(1, max_cycles + 1):
        x_start = x.copy()
        f_start = f_x
        gains, alphas = [], []

        for i in range(2):
            x_new, f_new, alpha = line_minimize(oracle, x, directions[i], stats=stats)
            gains.append(f_x - f_new)
            alphas.append(alpha)
            x, f_x = x_new, f_new

        shift = x - x_start
        norm = float(np.linalg.norm(shift))
        rows.append({"cycle": cycle, "x": x.copy(), "f": f_x, "shift": norm,
                     "alpha1": alphas[0], "alpha2": alphas[1], "n_f": oracle.n_f})

        if norm < tol or abs(f_start - f_x) <= 1e-12 * (1.0 + abs(f_x)):
            break

        x, f_x, _ = line_minimize(oracle, x, shift / norm, stats=stats)
        directions[int(np.argmax(gains))] = shift / norm
        if cycle % 3 == 0:
            directions = [np.array([1.0, 0.0]), np.array([0.0, 1.0])]

    return {"x": x, "f": f_x, "cycles": cycle, "n_f": oracle.n_f, "rows": rows,
            "line_stats": stats}
