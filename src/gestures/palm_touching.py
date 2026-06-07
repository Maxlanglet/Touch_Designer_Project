import time

import numpy as np

from src.geometry.normalized_quad import NormalizedQuad
from src.gestures.debounce import Debouncer, PointSmoother
from src.gestures.gesture_manager import DetectedGesture, GestureDetector, GestureType
from src.models.hand_data import HandData
from src.models.sim_normalized_landmarks import SimNormalizedLandmark

from .registry import register_gesture


@register_gesture("palm_touching")
class PalmTouchingDetector(GestureDetector):
    def __init__(
        self,
        hold_since: float = 0.12,
        grace_s: float = 0.25,
        smoothing_alpha: float = 0.5,
        palm_touching_threshold: float = 0.03,
    ):
        self.active = False
        self.was_triggering = False
        self.hold_since = hold_since
        self.grace_s = grace_s
        self.smoothing_alpha = smoothing_alpha
        self._debounce = Debouncer(hold_since)
        self._smoother = PointSmoother(smoothing_alpha)
        self.last_regions = None
        self.last_hands = []
        self.last_seen = 0.0
        self.palm_touching_threshold = palm_touching_threshold

    def reset(self):
        self.active = False
        self.was_triggering = False
        self.last_regions = None

    def detect(self, hands: list[HandData]) -> DetectedGesture:
        now = time.monotonic()

        raw = False
        if len(hands) == 2:
            left, right = sorted(hands, key=lambda h: h.hand_index)
            left_finger_tips = left.get_finger_tips()
            right_finger_tips = right.get_finger_tips()
            raw = self._palm_touching(left_finger_tips, right_finger_tips)

            self.last_regions = [
                NormalizedQuad.from_points(
                    [(finger.x, finger.y, finger.z) for finger in left_finger_tips]
                ),
                NormalizedQuad.from_points(
                    [(finger.x, finger.y, finger.z) for finger in right_finger_tips]
                ),
            ]
            self.last_hands = [left.hand_index, right.hand_index]
            self.last_seen = now

        stable = self._debounce.update(raw, now)
        if stable and not self.was_triggering:
            self.active = not self.active
        self.was_triggering = stable

        if not self.active:
            return None
        if not self.last_regions or (now - self.last_seen > self.grace_s):
            return None
        return DetectedGesture(
            type=GestureType.PALM_TOUCHING,
            hands=self.last_hands,
            regions=self.last_regions,
        )

    def _palm_touching(
        self,
        left_palm_points: list[tuple[float, float, float]],
        right_palm_points: list[tuple[float, float, float]],
    ) -> bool:
        all_touching = all(
            self.compute_3d_distance(left_palm_point, right_palm_point)
            < self.palm_touching_threshold
            for left_palm_point, right_palm_point in zip(
                left_palm_points, right_palm_points, strict=True
            )
        )
        return all_touching

    def compute_3d_distance(
        self, left_palm_point: SimNormalizedLandmark, right_palm_point: SimNormalizedLandmark
    ) -> float:
        return np.sqrt(
            (left_palm_point.x - right_palm_point.x) ** 2
            + (left_palm_point.y - right_palm_point.y) ** 2
            + (left_palm_point.z - right_palm_point.z) ** 2
        )
