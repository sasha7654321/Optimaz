import math

TAU = (math.sqrt(5.0) - 1.0) / 2.0        # 0.6180339887...


def dichotomy(oracle, a, b, eps=1e-5, delta=None, max_iter=500):
    """Метод дихотомії: два звернення до оракула на ітерацію."""
    if delta is None:
        delta = eps / 3.0
    rows = []
    k = 0
    while (b - a) > eps and k < max_iter:
        x1 = (a + b - delta) / 2.0
        x2 = (a + b + delta) / 2.0
        f1 = oracle.value(x1)
        f2 = oracle.value(x2)

        if f1 <= f2:
            decision = "f(x1) <= f(x2) -> b = x2"
            b = x2
        else:
            decision = "f(x1) >  f(x2) -> a = x1"
            a = x1

        k += 1
        rows.append({"k": k, "a": a, "b": b, "x1": x1, "x2": x2,
                     "f1": f1, "f2": f2, "L": b - a, "decision": decision})

    x_best = (a + b) / 2.0
    return {"x": x_best, "f": oracle.value(x_best), "a": a, "b": b,
            "iterations": k, "n_f": oracle.n_f, "rows": rows}


def golden_section(oracle, a, b, eps=1e-5, max_iter=500):
    """Золотий переріз: одне звернення на ітерацію (одна точка успадковується)."""
    x1 = b - TAU * (b - a)
    x2 = a + TAU * (b - a)
    f1 = oracle.value(x1)
    f2 = oracle.value(x2)

    rows = []
    k = 0
    while (b - a) > eps and k < max_iter:
        if f1 <= f2:
            decision = "f(x1) <= f(x2) -> b = x2, успадкували x1"
            b = x2
            x2, f2 = x1, f1
            x1 = b - TAU * (b - a)
            f1 = oracle.value(x1)
            new_point = x1
        else:
            decision = "f(x1) >  f(x2) -> a = x1, успадкували x2"
            a = x1
            x1, f1 = x2, f2
            x2 = a + TAU * (b - a)
            f2 = oracle.value(x2)
            new_point = x2

        k += 1
        rows.append({"k": k, "a": a, "b": b, "x1": x1, "x2": x2,
                     "f1": f1, "f2": f2, "L": b - a,
                     "new": new_point, "decision": decision})

    x_best = (a + b) / 2.0
    return {"x": x_best, "f": oracle.value(x_best), "a": a, "b": b,
            "iterations": k, "n_f": oracle.n_f, "rows": rows}
