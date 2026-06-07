from dataclasses import dataclass


@dataclass
class NormalizedBox:
    x1: float
    y1: float
    x2: float
    y2: float

    @property
    def width(self):
        return self.x2 - self.x1

    @property
    def height(self):
        return self.y2 - self.y1

    @property
    def center(self):
        return (self.x1 + self.x2) / 2, (self.y1 + self.y2) / 2

    def to_px(self, w: int, h: int):
        return (int(self.x1 * w), int(self.y1 * h)), (int(self.x2 * w), int(self.y2 * h))
