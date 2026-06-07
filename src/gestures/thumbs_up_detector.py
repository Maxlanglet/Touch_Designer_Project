from src.gestures.gesture_manager import DetectedGesture, GestureDetector, GestureType
from src.models.hand_data import HandData

from .registry import register_gesture


@register_gesture("thumb_up")
class ThumbsUpDetector(GestureDetector):
    def __init__(self):
        pass

    def detect(self, hands: list[HandData]) -> DetectedGesture:
        thumb_up = self.is_thumb_up(hands)
        if thumb_up:
            return DetectedGesture(
                type=GestureType.THUMB_UP,
                hands=[h.hand_index for h in hands],
            )
        return None

    def is_thumb_up(self, hands: list[HandData]):
        for hand in hands:
            thumb_landmark = hand.get_hand_stats()["gesture"]
            # Gesture name from the mediapipe gesture
            if thumb_landmark == "Thumb_Up":
                return True
        return False
