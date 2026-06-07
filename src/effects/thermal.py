import cv2
import numpy as np

from src.effects.effect import Effect
from src.effects.registry import register_effect
from src.geometry.normalized_box import NormalizedBox
from src.geometry.normalized_quad import NormalizedQuad
from src.renderer.colors import GREEN
from src.renderer.renderer import Renderer


@register_effect("thermal")
class Thermal(Effect):
    def __init__(self, render_outline: bool = False):
        super().__init__("thermal", render_outline)

    def apply(
        self, frame: np.ndarray, region: NormalizedBox | NormalizedQuad, renderer: Renderer
    ) -> np.ndarray:
        roi, x1, y1, x2, y2, pts = self.get_roi(frame, region)
        if roi is None:
            return frame

        local_pts = pts - np.array([x1, y1])
        mask = np.zeros(roi.shape[:2], dtype=np.uint8)
        cv2.fillConvexPoly(mask, local_pts, 255)

        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        thermal = cv2.applyColorMap(gray, cv2.COLORMAP_HOT)
        roi = cv2.copyTo(thermal, mask, roi)
        frame[y1:y2, x1:x2] = roi

        if self.render_outline:
            frame = renderer.render_polygon(frame, region.corners, GREEN, thickness=2)
        return frame
