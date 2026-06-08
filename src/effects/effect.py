import inspect
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import ClassVar

import numpy as np

from src.geometry.normalized_box import NormalizedBox
from src.geometry.normalized_quad import NormalizedQuad
from src.models.hand_data import HandData
from src.models.sim_normalized_landmarks import SimNormalizedLandmark
from src.renderer.renderer import Renderer


@dataclass
class RenderTarget:
    region: NormalizedBox | NormalizedQuad | None = None
    point: SimNormalizedLandmark | None = None
    depth: float = 0.0


class TargetKind(Enum):
    REGION = "region"
    POINT = "point"


def param_specs(cls) -> list[tuple[str, type, object]]:
    out = []
    for name, param in inspect.signature(cls.__init__).parameters.items():
        if name == "self" or param.default is inspect.Parameter.empty:
            continue
        typ = type(param.default) if param.default is not inspect.Parameter.empty else str
        out.append((name, typ, param.default))
    return out


def current_params(effect) -> dict[str, tuple[type, any]]:
    return {
        name: getattr(effect, name)
        for name, _, _ in param_specs(type(effect))
        if hasattr(effect, name)
    }


class Effect(ABC):
    # Default target kind to emit regions
    target_kind: ClassVar[set[TargetKind]] = {TargetKind.REGION}

    def __init__(self, name: str, render_outline: bool = False):
        self.name = name
        self.render_outline = render_outline

    @abstractmethod
    def apply(
        self,
        frame: np.ndarray,
        target: RenderTarget,
        renderer: Renderer,
        hands: list[HandData] | None = None,
    ) -> np.ndarray:
        """Applies the effect to the frame"""

    def get_roi(
        self, frame: np.ndarray, region: NormalizedBox | NormalizedQuad
    ) -> tuple[np.ndarray, int, int, int, int]:
        h, w, _ = frame.shape
        pts = region.to_px(w, h)

        x1, y1 = np.clip(pts.min(axis=0), [0, 0], [w - 1, h - 1])
        x2, y2 = np.clip(pts.max(axis=0), [0, 0], [w, h])
        if x2 <= x1 or y2 <= y1:
            return None, 0, 0, 0, 0, pts
        return frame[y1:y2, x1:x2], x1, y1, x2, y2, pts

    def get_region(self, target: RenderTarget) -> NormalizedBox | NormalizedQuad:
        if target.region:
            return target.region
        elif target.point:
            return NormalizedQuad(
                points=[target.point],
                normalized_points=[target.point],
            )
        else:
            return None
