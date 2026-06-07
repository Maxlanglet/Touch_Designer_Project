from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum

from src.geometry.normalized_box import NormalizedBox
from src.geometry.normalized_quad import NormalizedQuad
from src.models.hand_data import HandData


class GestureType(Enum):
    TWO_HAND_PINCH_SQUEEZE = "two_hand_pinch_squeeze"
    THUMB_UP = "thumb_up"
    CLOSED_FISTS = "closed_fists"


@dataclass
class DetectedGesture:
    type: GestureType
    hands: list[int]  # which hands are involved in the gesture
    confidence: float = 1.0
    region: NormalizedBox | NormalizedQuad | None = None
    regions: list[NormalizedQuad | NormalizedBox] = field(default_factory=list)
    metrics: dict = field(
        default_factory=dict
    )  # ex: {squeeze_distance: float, pinch_distances: list[float]}


class GestureDetector(ABC):
    @abstractmethod
    def detect(self, hands: list[HandData]) -> DetectedGesture:
        """Returns a DetectedGesture object if a gesture is detected, otherwise returns None"""

    @abstractmethod
    def reset(self):
        """
        Resets the gesture detector to its initial state
        """


class GestureManager:
    def __init__(self, detectors: list[GestureDetector]):
        self.detectors = detectors

    def reset_all(self):
        """
        Resets all gesture detectors to their initial state
        """
        for detector in self.detectors:
            detector.reset()

    def detect(self, hands: list[HandData]) -> list[DetectedGesture]:
        detected_gestures = []
        for detector in self.detectors:
            detected_gesture = detector.detect(hands)
            if detected_gesture:
                detected_gestures.append(detected_gesture)
        return detected_gestures
