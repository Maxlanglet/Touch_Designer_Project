import random

import cv2
import numpy as np

from src.effects.effect import Effect, RenderTarget
from src.effects.registry import EFFECT_REGISTRY, register_effect
from src.effects.web import Web
from src.geometry.normalized_box import NormalizedBox
from src.geometry.normalized_quad import NormalizedQuad
from src.models.hand_data import HandData
from src.models.sim_normalized_landmarks import SimNormalizedLandmark
from src.renderer.renderer import Renderer


@register_effect("mosaic")
class Mosaic(Web):
    def __init__(
        self,
        allowed_effects: list[str],
        render_outline: bool = False,
        connect_across: bool = False,
        connect_between: bool = False,
        thickness: int = 1,
        color: str = "WHITE",
        count: int = 5,
    ):
        super().__init__(
            render_outline=render_outline,
            connect_across=connect_across,
            connect_between=connect_between,
            thickness=thickness,
            color=color,
            count=count,
        )
        self.name = "mosaic"
        self.allowed_effects = [EFFECT_REGISTRY[effect]() for effect in (allowed_effects or [])]
        self.assigned_landmarks = self._assigned_landmarks()
        self._effect_by_key = {}

    def _deduce_regions_from_assigned_landmarks(
        self, landmarks: list[SimNormalizedLandmark]
    ) -> int:
        return len(self.assigned_landmarks)

    def _effect_for(self, key: int) -> Effect:
        if not self.allowed_effects:
            return None
        if key not in self._effect_by_key:
            self._effect_by_key[key] = random.choice(self.allowed_effects)
        return self._effect_by_key[key]

    def apply(
        self,
        frame: np.ndarray,
        target: RenderTarget,
        renderer: Renderer,
        hands: list[HandData] | None = None,
    ) -> np.ndarray:
        hands = hands or []

        if not hands:
            return frame

        h, w = frame.shape[:2]
        color = self._resolve_color(self.color)
        regions = self.get_regions_from_hands(hands, w, h)
        for key, region in regions:
            effect = self._effect_for(key)
            if effect is not None:
                frame = self.apply_effect_from_region(frame, effect, region, renderer)
        if self.render_outline:
            for _, region in regions:
                frame = renderer.render_polygon(
                    frame, region.corners, color, thickness=self.thickness
                )
        return frame

    def apply_effect_from_region(
        self,
        frame: np.ndarray,
        effect: Effect,
        region: NormalizedBox | NormalizedQuad,
        renderer: Renderer,
    ) -> np.ndarray:
        return effect.apply(frame=frame, target=RenderTarget(region=region), renderer=renderer)

    def gather_points(self, hands: list[HandData]) -> list[tuple[int, list[SimNormalizedLandmark]]]:
        pts = []
        if self.connect_across and len(hands) >= 2:
            a, b = sorted(hands, key=lambda h: h.hand_index)[:2]
            for i, lm in enumerate(a.get_landmarks()):
                pts.append((("a", i), (lm.x, lm.y, lm.z)))
            for i, lm in enumerate(b.get_landmarks()):
                pts.append((("b", i), (lm.x, lm.y, lm.z)))
        else:
            for hand in hands:
                for i, lm in enumerate(hand.get_landmarks()):
                    pts.append((("hand", hand.hand_index, i), (lm.x, lm.y, lm.z)))
        return pts

    def triangulate(self, pts_norm, w, h):
        px = [(int(x * w), int(y * h)) for x, y, _ in pts_norm]
        index_of = {}
        for i, p in enumerate(px):
            index_of.setdefault(p, i)
        xs = [p[0] for p in px]
        ys = [p[1] for p in px]
        subdiv = cv2.Subdiv2D(
            (min(xs) - 1, min(ys) - 1, max(xs) - min(xs) + 3, max(ys) - min(ys) + 3)
        )

        for p in px:
            subdiv.insert((float(p[0]), float(p[1])))
        tris = []
        for t in subdiv.getTriangleList():
            verts = [(int(t[0]), int(t[1])), (int(t[2]), int(t[3])), (int(t[4]), int(t[5]))]
            if all(v in index_of for v in verts):
                tris.append(tuple(sorted(index_of[v] for v in verts)))
        return tris

    def get_regions_from_hands(self, hands, w, h):
        pts = self.gather_points(hands)
        if len(pts) < 3:
            return []
        out = []
        for tri in self.triangulate([p for _, p in pts], w, h):
            corners = [pts[i][1] for i in tri]
            key = ("tri", tuple(pts[k][0] for k in tri))
            out.append((key, NormalizedQuad.from_points(corners)))
        return out

    def reset(self):
        self._effect_by_key.clear()
        self.assigned_landmarks = self._assigned_landmarks()
