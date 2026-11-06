import math
import numpy as np

# Режим дзеркального відбиття країв для білатерального фільтра.
# "reflect":   -1 -> 1, -2 -> 2 (Python-лістинг практикуму; саме з ним збігаються контрольні числа варіантів)
# "symmetric": -1 -> 0, -2 -> 1 (так описано словами в тексті практикуму)
BILATERAL_PAD_MODE = "reflect"


def gaussian_blur(img, sigma):
    """Гаусове розмиття: два проходи (рядки, потім стовпці), краї — найближчий піксель."""
    if sigma <= 1e-6:
        return img.copy()

    r = int(4.0 * sigma + 0.5)
    t = np.arange(-r, r + 1, dtype=float)
    kernel = np.exp(-(t * t) / (2.0 * sigma * sigma))
    kernel /= kernel.sum()

    h, w = img.shape

    tmp = np.zeros_like(img)
    for k in range(-r, r + 1):
        cols = np.clip(np.arange(w) + k, 0, w - 1)
        tmp += kernel[k + r] * img[:, cols]

    out = np.zeros_like(img)
    for k in range(-r, r + 1):
        rows = np.clip(np.arange(h) + k, 0, h - 1)
        out += kernel[k + r] * tmp[rows, :]

    return out


def gaussian_kernel(sigma):
    r = int(4.0 * sigma + 0.5)
    t = np.arange(-r, r + 1, dtype=float)
    k = np.exp(-(t * t) / (2.0 * sigma * sigma))
    return k / k.sum()


def bilateral_filter(img, sigma_spatial, sigma_range):
    """Білатеральний фільтр: усереднює лише схожих за яскравістю сусідів."""
    sigma_spatial = max(float(sigma_spatial), 1e-3)
    sigma_range = max(float(sigma_range), 1e-3)
    r = int(min(max(round(2.0 * sigma_spatial), 1), 5))

    h, w = img.shape
    padded = np.pad(img, r, mode=BILATERAL_PAD_MODE)
    num = np.zeros_like(img)
    den = np.zeros_like(img)

    for dy in range(-r, r + 1):
        for dx in range(-r, r + 1):
            shifted = padded[r + dy:r + dy + h, r + dx:r + dx + w]
            w_s = math.exp(-(dx * dx + dy * dy) / (2.0 * sigma_spatial ** 2))
            w_r = np.exp(-((shifted - img) ** 2) / (2.0 * sigma_range ** 2))
            weight = w_s * w_r
            num += weight * shifted
            den += weight

    return num / den
