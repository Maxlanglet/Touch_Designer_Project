import numpy as np

from src.effects.effect_manager import EffectManager
from src.gestures.gesture_manager import DetectedGesture, GestureType
from src.renderer.gesture_renderer import GestureRenderer
from src.renderer.hand_renderer import HandRenderer
from src.renderer.renderer import Renderer
from src.service.detector import Detector


class FramePipeline:
    def __init__(self, detector: Detector, renderer: Renderer, effect_manager: EffectManager):
        self.detector = detector
        self.renderer = renderer
        self.effect_manager = effect_manager

    def process(self, frame) -> np.ndarray:
        result = self.detector.detect(frame)

        if self._is_reset(result.gestures):
            self.detector.gesture_manager.reset_all()
            self.effect_manager.clear()

            gestures = []
        else:
            gestures = result.gestures
            self.effect_manager.handle(gestures)

        self.renderer.reset()
        for hand in result.hands:
            frame = HandRenderer(hand).render(frame, self.renderer)
        for gesture in result.gestures:
            frame = GestureRenderer(gesture).render(frame, self.renderer)

        frame = self.effect_manager.render(frame, self.renderer)
        return frame

    def _is_reset(self, gestures: list[DetectedGesture]) -> bool:
        return any(gesture.type == GestureType.THUMB_UP for gesture in gestures)
