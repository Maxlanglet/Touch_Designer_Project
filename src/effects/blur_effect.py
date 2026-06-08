import cv2
import numpy as np

from src.effects.effect import Effect, RenderTarget
from src.effects.registry import register_effect
from src.renderer.colors import GREEN
from src.renderer.renderer import Renderer


@register_effect("blur")
class BlurEffect(Effect):
    def __init__(self, render_outline: bool = False):
        super().__init__("blur", render_outline)

    def apply(self, frame: np.ndarray, target: RenderTarget, renderer: Renderer) -> np.ndarray:
        frame = self.box_blur(frame, target, renderer)
        return frame

    def box_blur(self, frame: np.ndarray, target: RenderTarget, renderer: Renderer) -> np.ndarray:
        region = self.get_region(target)
        if region is None:
            return frame

        roi, x1, y1, x2, y2, pts = self.get_roi(frame, region)
        if roi is None:
            return frame

        local_pts = pts - np.array([x1, y1])
        mask = np.zeros(roi.shape[:2], dtype=np.uint8)
        cv2.fillConvexPoly(mask, local_pts, 255)

        blurred = cv2.GaussianBlur(roi, (31, 31), 20)  # 0 => sigma derived from kernel
        roi = cv2.copyTo(blurred, mask, roi)
        frame[y1:y2, x1:x2] = roi
        if self.render_outline:
            frame = renderer.render_polygon(frame, region.corners, GREEN, thickness=2)
        return frame
