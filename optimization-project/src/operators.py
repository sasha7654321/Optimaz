import numpy as np
from scipy.fft import dctn, idctn   # єдине дозволене використання scipy: готове перетворення, не розв'язувач


def grad_xy(u):
    """Прямі різниці: праворуч і вниз."""
    gx = np.zeros_like(u)
    gx[:, :-1] = u[:, 1:] - u[:, :-1]
    gy = np.zeros_like(u)
    gy[:-1, :] = u[1:, :] - u[:-1, :]
    return gx, gy


def divergence(px, py):
    """Оператор, спряжений до градієнта зі знаком мінус."""
    d = np.zeros_like(px)
    d[:, 0] += px[:, 0]
    d[:, 1:-1] += px[:, 1:-1] - px[:, :-2]
    d[:, -1] -= px[:, -2]
    d[0, :] += py[0, :]
    d[1:-1, :] += py[1:-1, :] - py[:-2, :]
    d[-1, :] -= py[-2, :]
    return d


def laplacian(u):
    return divergence(*grad_xy(u))


def energy(I, noisy, lam):
    """E(I) = 1/2 ||I - I_шумне||^2 + lam/2 ||grad I||^2."""
    gx, gy = grad_xy(I)
    return 0.5 * float(np.sum((I - noisy) ** 2)) + 0.5 * lam * float(np.sum(gx * gx + gy * gy))


def energy_grad(I, noisy, lam):
    """grad E(I) = (I - I_шумне) - lam * Laplacian(I)."""
    return (I - noisy) - lam * laplacian(I)


def tikhonov_exact(noisy, lam):
    """Точний мінімум енергії Тихонова через косинусне перетворення."""
    h, w = noisy.shape
    i = np.arange(h)[:, None]
    j = np.arange(w)[None, :]
    mu = (2 - 2 * np.cos(np.pi * i / h)) + (2 - 2 * np.cos(np.pi * j / w))
    spectrum = dctn(noisy, type=2, norm="ortho")
    return idctn(spectrum / (1 + lam * mu), type=2, norm="ortho")


def check_adjoint(seed=0, shape=(9, 7)):
    """Перевірка 1: <grad u, p> = -<u, div p>."""
    rng = np.random.default_rng(seed)
    u = rng.normal(size=shape)
    px = rng.normal(size=shape)
    py = rng.normal(size=shape)
    gx, gy = grad_xy(u)
    left = float(np.sum(gx * px + gy * py))
    right = -float(np.sum(u * divergence(px, py)))
    return abs(left - right)


def check_gradient(noisy_small, lam, seed=0, t=1e-6):
    """Перевірка 2: похідна вздовж напрямку — скінченні різниці проти формули."""
    rng = np.random.default_rng(seed)
    I = rng.random(noisy_small.shape)
    V = rng.normal(size=noisy_small.shape)
    numeric = (energy(I + t * V, noisy_small, lam) - energy(I - t * V, noisy_small, lam)) / (2 * t)
    analytic = float(np.sum(energy_grad(I, noisy_small, lam) * V))
    return abs(numeric - analytic) / abs(analytic)
