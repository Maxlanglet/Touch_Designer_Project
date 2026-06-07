from dataclasses import dataclass
from enum import Enum

from src.gestures.gesture_manager import DetectedGesture, GestureType


class GestureEventType(Enum):
    STARTED = "started"
    ENDED = "ended"


@dataclass
class GestureEvent:
    type: GestureEventType
    gesture: GestureType


class GestureStateTracker:
    def __init__(self):
        self.active: set[GestureType] = set()

    def update(self, detected: list[DetectedGesture]) -> list[GestureEvent]:
        current = {g.type for g in detected}
        started = current - self.active
        ended = self.active - current
        self.active = current
        return [GestureEvent(GestureEventType.STARTED, g) for g in started] + [
            GestureEvent(GestureEventType.ENDED, g) for g in ended
        ]
