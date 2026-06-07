import time


class Debouncer:
    def __init__(self, delay: float = 0.12):
        self.delay = delay
        self.stable = False
        self._candidate = False
        self._since = 0.0

    def update(self, raw: bool, now: float | None = None) -> bool:
        now = now if now is not None else time.monotonic()
        if raw != self._candidate:
            self._candidate = raw
            self._since = now
        elif raw != self.stable and (now - self._since > self.delay):
            self.stable = raw
        return self.stable


class PointSmoother:
    def __init__(self, alpha: float = 0.5):
        self.alpha = alpha
        self._state: dict = {}

    def smooth(self, key, x, y, z):
        prev = self._state.get(key, None)
        if prev is None:
            cur = (x, y, z)
        else:
            a = self.alpha
            cur = (a * x + (1 - a) * prev[0], a * y + (1 - a) * prev[1], a * z + (1 - a) * prev[2])
        self._state[key] = cur
        return cur
