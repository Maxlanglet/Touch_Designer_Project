import time

import numpy as np

from src.geometry.normalized_quad import NormalizedQuad
from src.gestures.debounce import Debouncer, PointSmoother
from src.gestures.gesture_manager import DetectedGesture, GestureDetector, GestureType
from src.models.hand_data import HandData
from src.models.sim_normalized_landmarks import SimNormalizedLandmark

from .registry import register_gesture

PINCH_NORMALIZED_THRESHOLD = 0.037
SQUEEZE_NORMALIZED_THRESHOLD = 0.037


@register_gesture("two_hand_pinch_squeeze")
class TwoHandPinchSqueezeGestureDetector(GestureDetector):
    def __init__(
        self,
        pinch_normalized_threshold: float = PINCH_NORMALIZED_THRESHOLD,
        squeeze_normalized_threshold: float = SQUEEZE_NORMALIZED_THRESHOLD,
        hold_since: float = 0.12,
        grace_s: float = 0.25,
        smoothing_alpha: float = 0.5,
    ):
        self.pinch_normalized_threshold = pinch_normalized_threshold
        self.squeeze_normalized_threshold = squeeze_normalized_threshold
        self.hold_since = hold_since
        self.grace_s = grace_s
        self.smoothing_alpha = smoothing_alpha
        self.active = False
        self.was_triggering = False
        self.debounce = Debouncer(hold_since)
        self.smoother = PointSmoother(smoothing_alpha)
        self.last_regions = None
        self.last_hands = []
        self.last_seen = 0.0
        self.metrics = {}

    def reset(self):
        self.active = False
        self.was_triggering = False
        self.last_regions = None

    def detect(self, hands: list[HandData]) -> DetectedGesture:
        now = time.monotonic()
        self.debounce.hold_s = self.hold_since
        self.smoother.alpha = self.smoothing_alpha

        raw = False
        if len(hands) == 2:
            index_thumb_landmarks = self._get_index_thumb_landmarks(hands)
            pinched, squeezed, sd, pd = self.pinched_and_squeezed(hands, index_thumb_landmarks)
            raw = pinched and squeezed
            self.last_regions = self._quad_from_fingertips(self._get_index_thumb_landmarks(hands))
            self.last_hands = [h.hand_index for h in hands]
            self.last_seen = now
            self.metrics = {"squeeze_distance": sd, "pinch_distance": pd}

        stable = self.debounce.update(raw, now)
        if stable and not self.was_triggering:
            self.active = not self.active
        self.was_triggering = stable

        if not self.active:
            return None
        if not self.last_regions or (now - self.last_seen > self.grace_s):
            return None

        return DetectedGesture(
            type=GestureType.TWO_HAND_PINCH_SQUEEZE,
            hands=self.last_hands,
            regions=[self.last_regions],
            metrics=self.metrics,
        )

    def _quad_from_fingertips(self, index_thumb_landmarks) -> NormalizedQuad:
        points = []
        for index, thumb in index_thumb_landmarks:
            points.append((index.x, index.y, index.z))
            points.append((thumb.x, thumb.y, thumb.z))
        return NormalizedQuad.from_points(points)

    def _quad_from_fingertips(
        self, per_hand_fingertips: list[tuple[SimNormalizedLandmark, SimNormalizedLandmark]]
    ) -> NormalizedQuad:
        points = []
        for hand_i, (thumb, index) in enumerate(per_hand_fingertips):
            points.append(self.smoother.smooth((hand_i, "thumb"), thumb.x, thumb.y, thumb.z))
            points.append(self.smoother.smooth((hand_i, "index"), index.x, index.y, index.z))
        return NormalizedQuad.from_points(points)

    def _get_index_thumb_landmarks(self, hand_data: list[HandData]):
        index_thumb_landmarks = []
        for hand in hand_data:
            index_finger_landmark = hand.get_index_finger_landmark()
            thumb_finger_landmark = hand.get_thumb_finger_landmark()
            index_thumb_landmarks.append((index_finger_landmark, thumb_finger_landmark))
        return index_thumb_landmarks

    def pinched_and_squeezed(
        self,
        hand_data: list[HandData],
        index_thumb_landmarks: list[tuple[SimNormalizedLandmark, SimNormalizedLandmark]],
    ):
        if len(index_thumb_landmarks) <= 1:
            return False, False, 0, 0

        if len(index_thumb_landmarks) > 2:
            return False, False, 0, 0

        # 3 conditions to be met: both hands index touching and thumb touching, and both hands touching each other

        pinch_distances = []
        average_coordinates_intra = []
        for index_thumb_landmark in index_thumb_landmarks:
            pinch_distances.append(
                self.compute_2d_distance(index_thumb_landmark[0], index_thumb_landmark[1])
            )
            average_coordinates_intra.append(self.compute_average_coordinates(index_thumb_landmark))
        is_squeezing, squeeze_distance = self.is_squeezing(
            average_coordinates_intra[0], average_coordinates_intra[1]
        )

        is_pinching, pinch_distance = self.is_pinching(pinch_distances)

        return is_pinching, is_squeezing, squeeze_distance, pinch_distance

    def is_squeezing(self, landmark1, landmark2):
        distance = self.compute_2d_distance(landmark1, landmark2)
        return distance < self.squeeze_normalized_threshold, distance

    def is_pinching(self, pinch_distances):
        mean_distance = np.mean(pinch_distances)
        return all(
            pinch_distance < self.pinch_normalized_threshold for pinch_distance in pinch_distances
        ), mean_distance

    def compute_average_coordinates(self, landmarks):
        average_x = np.mean([landmark.x for landmark in landmarks])
        average_y = np.mean([landmark.y for landmark in landmarks])
        average_z = np.mean([landmark.z for landmark in landmarks])
        return SimNormalizedLandmark(x=average_x, y=average_y, z=average_z)

    def compute_2d_distance(self, landmark1, landmark2):
        return np.sqrt((landmark1.x - landmark2.x) ** 2 + (landmark1.y - landmark2.y) ** 2)
