import cv2
import numpy as np

from src.effects.effect import Effect
from src.effects.registry import register_effect
from src.geometry.normalized_box import NormalizedBox
from src.geometry.normalized_quad import NormalizedQuad
from src.renderer.colors import GREEN
from src.renderer.renderer import Renderer


@register_effect("pixelate")
class Pixelate(Effect):
    def __init__(self, render_outline: bool = False, block_size: int = 10):
        super().__init__("pixelate", render_outline)
        self.block_size = block_size

    def apply(
        self, frame: np.ndarray, region: NormalizedBox | NormalizedQuad, renderer: Renderer
    ) -> np.ndarray:
        frame = self.pixelate(frame, region, renderer, block_size=self.block_size)
        return frame

    def pixelate(
        self,
        frame: np.ndarray,
        region: NormalizedBox | NormalizedQuad,
        renderer: Renderer,
        block_size: int = 10,
    ) -> np.ndarray:
        roi, x1, y1, x2, y2, pts = self.get_roi(frame, region)
        if roi is None:
            return frame
        h, w = roi.shape[:2]

        # Guard against invalid block sizes and tiny ROIs.
        safe_block = max(1, block_size)

        # Smaller intermediate image -> stronger pixelation.
        small_w = max(1, w // safe_block)
        small_h = max(1, h // safe_block)

        local_pts = pts - np.array([x1, y1])
        mask = np.zeros(roi.shape[:2], dtype=np.uint8)
        cv2.fillConvexPoly(mask, local_pts, 255)

        temp = cv2.resize(roi, (small_w, small_h), interpolation=cv2.INTER_LINEAR)
        pixelated = cv2.resize(temp, (w, h), interpolation=cv2.INTER_NEAREST)

        roi = cv2.copyTo(pixelated, mask, roi)
        frame[y1:y2, x1:x2] = roi

        if self.render_outline:
            frame = renderer.render_polygon(frame, region.corners, GREEN, thickness=2)
        return frame
