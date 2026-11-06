import math
import numpy as np


def mse(a, b):
    """Середньоквадратична помилка."""
    return float(np.mean((a - b) ** 2))


def psnr(a, b):
    """PSNR у децибелах (пік = 1)."""
    value = mse(a, b)
    if value <= 0.0:
        return float("inf")
    return 10.0 * math.log10(1.0 / value)
