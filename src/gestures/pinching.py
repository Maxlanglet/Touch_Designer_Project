import time

import numpy as np

from src.effects.effect import TargetKind
from src.gestures.debounce import Debouncer, PointSmoother
from src.gestures.gesture_manager import DetectedGesture, GestureDetector, GestureType
from src.models.hand_data import HandData
from src.models.sim_normalized_landmarks import SimNormalizedLandmark

from .registry import register_gesture


@register_gesture("pinching")
class PinchingDetector(GestureDetector):
    # Overrides the default target kind to emit points instead of regions
    emits = TargetKind.POINT

    def __init__(
        self,
        trigger_hold_since: float = 0.6,
        hold_since: float = 0.12,
        grace_s: float = 0.25,
        smoothing_alpha: float = 0.5,
        pinch_normalized_threshold: float = 0.03,
    ):
        self.active = False
        self.trigger_hold_since = trigger_hold_since
        self.hold_since = hold_since
        self._held_since = None
        self.smoothing_alpha = smoothing_alpha
        self._debounce = Debouncer(hold_since)
        self._smoother = PointSmoother(smoothing_alpha)
        self.last_hands = []
        self.last_seen = 0.0
        self.metrics = {}
        self.pinch_normalized_threshold = pinch_normalized_threshold

    def reset(self):
        self._debounce = Debouncer(self.hold_since)
        self.active = False
        self.last_hands = []
        self.last_seen = 0.0
        self.metrics = {}

    def detect(self, hands: list[HandData]) -> DetectedGesture:
        now = time.monotonic()
        self._debounce.delay = self.hold_since

        points, hand_ids = self._pinching_points(hands)
        stable = self._debounce.update(bool(points), now)

        if stable:
            if self._held_since is None:
                self._held_since = now
            held_for = now - self._held_since
        else:
            self._held_since = None
            held_for = 0.0

        self.active = stable and held_for > self.trigger_hold_since

        progress = (
            min(held_for / self.trigger_hold_since, 1.0) if self.trigger_hold_since > 0 else 1.0
        )
        self.metrics = {"hold_progress": progress}

        if not self.active or not points:
            return None

        self.last_hands = hand_ids

        return DetectedGesture(
            type=GestureType.PINCHING,
            hands=hand_ids,
            points=points,
            metrics=self.metrics,
        )

    def _pinching_points(self, hands: list[HandData]) -> list[SimNormalizedLandmark]:
        points, ids = [], []
        for hand in hands:
            thumb, index = hand.get_pinch_fingertips()
            if self._dist(thumb, index) < self.pinch_normalized_threshold:
                points.append(
                    SimNormalizedLandmark(
                        x=(thumb.x + index.x) / 2,
                        y=(thumb.y + index.y) / 2,
                        z=(thumb.z + index.z) / 2,
                    )
                )
                ids.append(hand.hand_index)
        return points, ids

    @staticmethod
    def _dist(landmark1, landmark2):
        return np.sqrt((landmark1.x - landmark2.x) ** 2 + (landmark1.y - landmark2.y) ** 2)
