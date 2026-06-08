import cv2
import numpy as np

from src.effects.effect import Effect, RenderTarget
from src.effects.registry import register_effect
from src.models.hand_data import HandData
from src.renderer.colors import GREEN
from src.renderer.renderer import Renderer


@register_effect("canny_edges")
class CannyEdges(Effect):
    def __init__(self, render_outline: bool = False, threshold1: int = 100, threshold2: int = 200):
        super().__init__("canny_edges", render_outline)
        self.threshold1 = threshold1
        self.threshold2 = threshold2

    def apply(
        self,
        frame: np.ndarray,
        target: RenderTarget,
        renderer: Renderer,
        hands: list[HandData] | None = None,
    ) -> np.ndarray:
        frame = self.canny_edges(frame, target, renderer)
        return frame

    def canny_edges(
        self, frame: np.ndarray, target: RenderTarget, renderer: Renderer
    ) -> np.ndarray:
        region = self.get_region(target)
        if region is None:
            return frame

        roi, x1, y1, x2, y2, pts = self.get_roi(frame, region)
        if roi is None:
            return frame

        local_pts = pts - np.array([x1, y1])
        mask = np.zeros(roi.shape[:2], dtype=np.uint8)
        cv2.fillConvexPoly(mask, local_pts, 255)

        edges = cv2.Canny(roi, self.threshold1, self.threshold2)
        edges = cv2.convertScaleAbs(edges)
        edges = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
        roi = cv2.copyTo(edges, mask, roi)

        frame[y1:y2, x1:x2] = roi

        if self.render_outline:
            frame = renderer.render_polygon(frame, region.corners, GREEN, thickness=2)
        return frame
