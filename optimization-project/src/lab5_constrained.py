import numpy as np

from src.operators import energy, energy_grad


def project(I, lower=0.0, upper=1.0):
    """Проєкція на коробку [0,1]^n: найближча допустима точка."""
    return np.clip(I, lower, upper)


def violation(I, lower=0.0, upper=1.0):
    below = np.maximum(0.0, lower - I)
    above = np.maximum(0.0, I - upper)
    return {"max": float(max(below.max(), above.max())),
            "count": int(np.count_nonzero((I < lower) | (I > upper)))}


def projected_gradient(raw, lam, L, accelerated=False, max_iter=200, tol=1e-6,
                       stop="gmap"):
    """Проєкція градієнта (accelerated=True — FISTA).

    stop='gmap' — правильний критерій (градієнтне відображення),
    stop='grad' — НЕПРАВИЛЬНИЙ критерій ||grad E|| (для досліду 1).
    """
    h = 1.0 / L
    I = project(raw.copy())
    I_prev = I.copy()
    Y = I.copy()
    n_f = n_grad = 0
    rows = []
    converged = False

    for k in range(1, max_iter + 1):
        point = Y if accelerated else I
        g = energy_grad(point, raw, lam); n_grad += 1
        I_new = project(point - h * g)

        g_map = float(np.linalg.norm(I_new - point) / h)
        g_plain = float(np.linalg.norm(g))
        active = int(np.count_nonzero((I <= 1e-12) | (I >= 1.0 - 1e-12)))
        v = violation(I)
        e_k = energy(I, raw, lam); n_f += 1
        rows.append({"k": k - 1, "E": e_k, "G": g_map, "grad": g_plain, "active": active,
                     "violation": v["max"], "n_f": n_f, "n_grad": n_grad})

        crit = g_map if stop == "gmap" else g_plain
        if crit <= tol:
            converged = True
            break

        if accelerated:
            beta = (k - 1) / (k + 2)
            Y = I_new + beta * (I_new - I_prev)
            I_prev = I_new
        I = I_new

    return {"I": I, "iterations": len(rows) - 1, "n_f": n_f, "n_grad": n_grad,
            "E": energy(I, raw, lam), "G": rows[-1]["G"],
            "converged": converged, "rows": rows}


def penalty_value(I, lower=0.0, upper=1.0):
    below = np.maximum(0.0, lower - I)
    above = np.maximum(0.0, I - upper)
    return float(np.sum(below ** 2 + above ** 2))


def penalty_grad(I, lower=0.0, upper=1.0):
    return -2.0 * np.maximum(0.0, lower - I) + 2.0 * np.maximum(0.0, I - upper)


def penalty_method(raw, lam, L, r0=1.0, growth=10.0, outer=7,
                   inner=100, inner_tol=1e-6, tol=1e-8):
    """Зовнішні штрафи: послідовність задач без обмежень зі зростаючим r."""
    I = raw.copy()
    r = r0
    n_f = n_grad = 0
    rows = []

    for s in range(outer):
        L_r = L + 2.0 * r
        h = 1.0 / L_r
        I_prev = I.copy()

        for k in range(1, inner + 1):
            beta = (k - 1) / (k + 2)
            Y = I + beta * (I - I_prev)
            g = energy_grad(Y, raw, lam) + r * penalty_grad(Y); n_grad += 1
            I_new = Y - h * g
            if float(np.linalg.norm(g)) <= inner_tol:
                I_prev, I = I, I_new
                break
            I_prev, I = I, I_new

        P = penalty_value(I); n_f += 1
        v = violation(I)
        rows.append({"s": s, "r": r, "L_r": L_r, "step": h, "E": energy(I, raw, lam), "P": P,
                     "rP": r * P, "violation": v["max"], "outside": v["count"],
                     "n_f": n_f, "n_grad": n_grad})

        if r * P <= tol:
            break
        r *= growth

    return {"I": I, "n_f": n_f, "n_grad": n_grad,
            "E": energy(I, raw, lam), "rows": rows}
