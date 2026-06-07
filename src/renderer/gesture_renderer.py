import numpy as np

from src.gestures.gesture_manager import DetectedGesture
from src.renderer.colors import BLUE
from src.renderer.renderable import Renderable
from src.renderer.renderer import Renderer


class GestureRenderer(Renderable):
    def __init__(self, gesture: DetectedGesture):
        self.gesture = gesture

    def render(self, frame: np.ndarray, canvas: Renderer) -> np.ndarray:
        frame = canvas.text(frame, f"Gesture: {self.gesture.type}", BLUE)
        return frame
