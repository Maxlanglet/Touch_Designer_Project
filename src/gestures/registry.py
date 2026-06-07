from src.gestures.gesture_manager import GestureDetector

GESTURE_REGISTRY: dict[str, type[GestureDetector]] = {}


def register_gesture(name: str):
    def deco(cls):
        GESTURE_REGISTRY[name] = cls
        return cls

    return deco
