import itertools
from collections import deque
from typing import ClassVar

from src.effects.effect import Effect, TargetKind
from src.effects.registry import register_effect
from src.renderer.colors import _COLORS, WHITE


@register_effect("draw")
class Draw(Effect):
    target_kinds: ClassVar[set[TargetKind]] = {TargetKind.POINT}

    def __init__(self, color="GREEN", thickness=3, max_points=64):
        super().__init__("draw")
        self.color = color
        self.thickness = thickness
        self.max_points = max_points
        self._pts = deque(maxlen=max_points)

    def apply(self, frame, target, renderer, hands=None):
        if target is None or target.point is None:
            return frame
        self._pts.append(target.point)
        col = _COLORS.get(self.color, WHITE)
        pts = list(self._pts)
        for a, b in itertools.pairwise(pts):
            frame = renderer.line(frame, a, b, col, thickness=self.thickness)
        return frame
