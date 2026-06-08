import cv2
import numpy as np
from loguru import logger

from src.effects.effect import Effect, RenderTarget
from src.effects.registry import register_effect
from src.models.hand_data import HandData
from src.renderer.colors import GREEN
from src.renderer.renderer import Renderer


@register_effect("sobel_edges")
class SobelEdges(Effect):
    def __init__(self, render_outline: bool = False, kernel_size: int = 5):
        super().__init__("sobel_edges", render_outline)
        if kernel_size % 2 == 0:
            logger.warning(f"Kernel size must be odd. Using 5 instead. Got {kernel_size}")
            kernel_size = 5
        self.kernel_size = kernel_size

    def apply(
        self,
        frame: np.ndarray,
        target: RenderTarget,
        renderer: Renderer,
        hands: list[HandData] | None = None,
    ) -> np.ndarray:
        frame = self.edges(frame, target, renderer)
        return frame

    def edges(self, frame: np.ndarray, target: RenderTarget, renderer: Renderer) -> np.ndarray:
        region = self.get_region(target)
        if region is None:
            return frame

        roi, x1, y1, x2, y2, pts = self.get_roi(frame, region)
        if roi is None:
            return frame

        local_pts = pts - np.array([x1, y1])
        mask = np.zeros(roi.shape[:2], dtype=np.uint8)
        cv2.fillConvexPoly(mask, local_pts, 255)

        edges = cv2.Sobel(roi, cv2.CV_64F, 1, 1, ksize=self.kernel_size)
        edges = cv2.convertScaleAbs(edges)
        roi = cv2.copyTo(edges, mask, roi)
        frame[y1:y2, x1:x2] = roi

        if self.render_outline:
            frame = renderer.render_polygon(frame, region.corners, GREEN, thickness=2)
        return frame
