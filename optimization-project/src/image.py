import math
import numpy as np
from PIL import Image as PILImage

from src.rng import Rng


def clean_image(size=128):
    """Малює еталонне зображення: градієнт, фігури, шахівниця, смуги."""
    img = np.zeros((size, size), dtype=float)

    for i in range(size):
        for j in range(size):
            u = j / (size - 1)
            v = i / (size - 1)

            # 1. фон
            val = 0.25 + 0.35 * (0.6 * u + 0.4 * v)
            val += 0.08 * math.cos(2 * math.pi * math.sqrt((u - 0.5) ** 2 + (v - 0.5) ** 2))

            # 2. світле коло
            if (u - 0.32) ** 2 + (v - 0.34) ** 2 <= 0.16 ** 2:
                val = 0.92
            # 3. темний квадрат
            if abs(u - 0.70) <= 0.15 and abs(v - 0.30) <= 0.15:
                val = 0.10
            # 4. трикутник
            if 0.60 < v < 0.90 and abs(u - 0.30) < 0.9 * (v - 0.60):
                val = 0.68
            # 5. шахівниця
            if 0.58 < u < 0.92 and 0.60 < v < 0.90:
                cell = max(3, size // 12)
                val = 0.85 if ((j // cell) + (i // cell)) % 2 == 0 else 0.20

            img[i][j] = min(max(val, 0.0), 1.0)

    # 6. чотири білі смуги вгорі
    for idx in range(4):
        thickness = max(2, (4 - idx) * max(1, size // 64))
        row = int(0.04 * size) + idx * max(3, size // 22)
        if row + thickness < size:
            img[row:row + thickness, int(0.05 * size):int(0.95 * size)] = 0.98

    return img


def add_noise(clean, sigma, seed):
    """Додає гаусів шум. Повертає (raw, noisy): raw — без обрізання, noisy — у [0,1]."""
    rng = Rng(seed)
    h, w = clean.shape
    raw = np.zeros((h, w), dtype=float)
    for i in range(h):                       # порядок обходу важливий: рядок за рядком
        for j in range(w):
            raw[i][j] = clean[i][j] + sigma * rng.normal()
    noisy = np.clip(raw, 0.0, 1.0)
    return raw, noisy


def save_png(img, path):
    """Зберігає масив [0,1] у 8-бітний PNG."""
    data = (np.clip(img, 0.0, 1.0) * 255.0).round().astype(np.uint8)
    PILImage.fromarray(data, mode="L").save(path)


def load_png(path, size=128):
    with PILImage.open(path) as handle:
        gray = handle.convert("L").resize((size, size), PILImage.BILINEAR)
    return np.asarray(gray, dtype=float) / 255.0
