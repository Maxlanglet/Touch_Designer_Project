import time

from src.geometry.normalized_quad import NormalizedQuad
from src.gestures.debounce import Debouncer, PointSmoother
from src.gestures.gesture_manager import DetectedGesture, GestureDetector, GestureType
from src.models.hand_data import HandData

from .registry import register_gesture


@register_gesture("closed_fists")
class ClosedFistsDetector(GestureDetector):
    def __init__(
        self, hold_since: float = 0.12, grace_s: float = 0.25, smoothing_alpha: float = 0.5
    ):
        self.active = False
        self.was_triggering = False
        self.hold_since = hold_since
        self.grace_s = grace_s
        self.smoothing_alpha = smoothing_alpha
        self._debounce = Debouncer(hold_since)
        self._smoother = PointSmoother(smoothing_alpha)
        self.last_regions = []
        self.last_hands = []
        self.last_seen = 0.0

    def reset(self):
        self.active = False
        self.was_triggering = False
        self.last_regions = []

    def detect(self, hands: list[HandData]) -> DetectedGesture:
        now = time.monotonic()
        self._debounce.hold_s = self.hold_since
        self._smoother.alpha = self.smoothing_alpha

        both = len(hands) == 2
        raw = both and all(hand.gesture == "Closed_Fist" for hand in hands)
        if len(hands) != 2:
            return None

        if both:
            left, right = sorted(hands, key=lambda h: h.hand_index)
            regions = self.fingertip_quads(left, right)
            self.last_regions = regions
            self.last_hands = [h.hand_index for h in hands]
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
            type=GestureType.CLOSED_FISTS,
            hands=self.last_hands,
            regions=self.last_regions,
        )

    def fingertip_quads(self, left: HandData, right: HandData) -> list[NormalizedQuad]:
        lt = left.get_finger_tips()
        rt = right.get_finger_tips()
        ls = [
            self._smoother.smooth(("L", i), left_finger.x, left_finger.y, left_finger.z)
            for i, left_finger in enumerate(lt)
        ]
        rs = [
            self._smoother.smooth(("R", i), right_finger.x, right_finger.y, right_finger.z)
            for i, right_finger in enumerate(rt)
        ]

        quads = []
        for k in range(len(ls) - 1):
            corners = [
                ls[k],
                ls[k + 1],
                rs[k + 1],
                rs[k],
            ]
            quads.append(NormalizedQuad.from_points(corners))
        return quads
