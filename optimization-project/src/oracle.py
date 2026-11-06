class Oracle:
    """Обгортка навколо цільової функції, що рахує звернення."""

    def __init__(self, func):
        self.func = func
        self.n_f = 0
        self.history = []

    def value(self, x):
        self.n_f += 1
        v = float(self.func(x))
        self.history.append((x, v))
        return v

    def peek(self, x):
        """Обчислити без підрахунку — лише для графіків."""
        return float(self.func(x))

    def reset(self):
        self.n_f = 0
        self.history.clear()
