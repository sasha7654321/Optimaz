import numpy as np

from src.operators import energy, energy_grad


def gradient_method(noisy, lam, L, kind="gd", beta=None, step=None,
                    max_iter=200, tol=1e-6, x_ref=None):
    """Чотири методи в одній функції.

    kind: 'gd'     — спуск зі сталим кроком h = 1/L
          'armijo' — спуск із правилом Армійо
          'hb'     — важка кулька (Поляк), потрібне beta
          'nag'    — метод Нестерова (beta=None: опуклий, beta=число: сильно опуклий)
    """
    h = step if step is not None else 1.0 / L
    I = noisy.copy()
    I_prev = I.copy()
    n_f = n_grad = 0
    trial = h
    rows = []
    converged = False

    for k in range(1, max_iter + 1):
        if kind == "nag":
            b = (k - 1) / (k + 2) if beta is None else beta
            Y = I + b * (I - I_prev)
            g = energy_grad(Y, noisy, lam); n_grad += 1
            g_norm = float(np.linalg.norm(g))
            I_new = Y - h * g
        else:
            g = energy_grad(I, noisy, lam); n_grad += 1
            g_norm = float(np.linalg.norm(g))

            if kind == "armijo":
                s = trial
                f_I = energy(I, noisy, lam); n_f += 1
                g2 = float(np.sum(g * g))
                for _ in range(40):
                    candidate = I - s * g
                    f_c = energy(candidate, noisy, lam); n_f += 1
                    if f_c <= f_I - 1e-4 * s * g2:
                        break
                    s *= 0.5
                I_new = candidate
                trial = s * 1.5
                h = s
            elif kind == "hb":
                I_new = I - h * g + beta * (I - I_prev)
            else:
                I_new = I - h * g

        e_k = energy(I, noisy, lam); n_f += 1
        rows.append({"k": k - 1, "E": e_k, "grad": g_norm, "h": h,
                     "n_f": n_f, "n_grad": n_grad,
                     "dist": float(np.linalg.norm(I - x_ref)) if x_ref is not None else None})

        if g_norm <= tol:
            converged = True
            break

        I_prev, I = I, I_new

    return {"I": I, "iterations": len(rows) - 1, "n_f": n_f, "n_grad": n_grad,
            "E": energy(I, noisy, lam), "grad": rows[-1]["grad"],
            "converged": converged, "rows": rows}


def tv_denoise(noisy, lam_tv, eps_tv, n_iter=500):
    """Додаткове завдання: повна варіація (згладжена), метод Нестерова (опуклий варіант)."""
    from src.operators import grad_xy, divergence
    L = 1.0 + 8.0 * lam_tv / eps_tv
    h = 1.0 / L

    def grad(I):
        gx, gy = grad_xy(I)
        mag = np.sqrt(gx * gx + gy * gy + eps_tv ** 2)
        return (I - noisy) - lam_tv * divergence(gx / mag, gy / mag)

    I = noisy.copy()
    I_prev = I.copy()
    for k in range(1, n_iter + 1):
        b = (k - 1) / (k + 2)
        Y = I + b * (I - I_prev)
        I_new = Y - h * grad(Y)
        I_prev, I = I, I_new
    return I
