import math
from dataclasses import dataclass

import numpy as np


@dataclass
class NormalizedQuad:
    corners: list[tuple[float, float, float, float]]

    def to_px(self, w: int, h: int):
        return np.array([(int(corner[0] * w), int(corner[1] * h)) for corner in self.corners])

    @classmethod
    def from_points(cls, points):
        cx = sum(p[0] for p in points) / len(points)
        cy = sum(p[1] for p in points) / len(points)
        ordered = sorted(points, key=lambda p: math.atan2(p[1] - cy, p[0] - cx))
        return cls(ordered)

    @property
    def depth(self) -> float:
        return sum(c[2] if len(c) > 2 else 0 for c in self.corners) / len(self.corners)
