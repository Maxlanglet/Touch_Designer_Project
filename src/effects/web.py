import random

import numpy as np

from src.effects.effect import Effect
from src.effects.registry import register_effect
from src.geometry.normalized_box import NormalizedBox
from src.geometry.normalized_quad import NormalizedQuad
from src.models.hand_data import HandData
from src.renderer.colors import BLUE, GREEN, RED, WHITE
from src.renderer.renderer import Renderer

_COLORS = {"WHITE": WHITE, "RED": RED, "GREEN": GREEN, "BLUE": BLUE}


@register_effect("web")
class Web(Effect):
    def __init__(
        self,
        render_outline: bool = False,
        connect_across: bool = False,
        thickness: int = 1,
        color: str = "WHITE",
        count: int = 5,
    ):
        super().__init__("web", render_outline)
        self.connect_across = connect_across
        self.thickness = thickness
        self.color = color
        self.count = count
        self.assigned_count = None
        self.assigned_landmarks = []
        self.assigned_landmarks = self._assigned_landmarks()

    def _resolve_color(self, color: str) -> tuple[int, int, int]:
        return self._COLORS.get(color, WHITE)

    def apply(
        self,
        frame: np.ndarray,
        region: NormalizedBox | NormalizedQuad,
        renderer: Renderer,
        hands: list[HandData] | None = None,
    ) -> np.ndarray:
        hands = hands or []
        color = self._resolve_color(self.color)
        if not hands:
            return frame
        self._ensure_assigned()

        for hand in hands:
            landmarks = hand.get_landmarks()
            for i in range(len(landmarks)):
                for j in range(i + 1, len(landmarks)):
                    frame = renderer.line(
                        frame, landmarks[i], landmarks[j], color, thickness=self.thickness
                    )

        if self.connect_across and len(hands) >= 2:
            a, b = sorted(hands, key=lambda h: h.hand_index)[:2]

            a_landmarks = a.get_landmarks()
            for i, lma in enumerate(a_landmarks):
                b_landmarks = b.get_landmarks()
                for _, lmb in enumerate(self.assigned_landmarks[i]):
                    frame = renderer.line(
                        frame, lma, b_landmarks[lmb], color, thickness=self.thickness
                    )
        return frame

    def _ensure_assigned(self):
        if self.assigned_count != self.count:
            self.assigned_landmarks = self._assigned_landmarks()
            self.assigned_count = self.count

    def _assigned_landmarks(self):
        LANDMARKS_COUNT = 21
        k = max(0, min(self.count, LANDMARKS_COUNT))  # clamp: random.sample errors if k > 21
        return [random.sample(range(LANDMARKS_COUNT), k) for _ in range(LANDMARKS_COUNT)]
