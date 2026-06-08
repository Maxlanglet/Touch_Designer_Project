import threading

import numpy as np

from src.effects.effect import Effect, RenderTarget
from src.geometry.normalized_box import NormalizedBox
from src.geometry.normalized_quad import NormalizedQuad
from src.gestures.gesture_manager import DetectedGesture, GestureType
from src.models.hand_data import HandData
from src.renderer.renderer import Renderer


class EffectManager:
    def __init__(self, bindings: dict[GestureType, list[Effect]]):
        self.bindings = bindings
        self.assignements: dict[GestureType, list[Effect]] = {}
        self.active: dict[GestureType, NormalizedBox | NormalizedQuad] = {}
        self._lock = threading.Lock()
        self.hands: list[HandData] | None = None

    def handle(self, gestures, hands: list[HandData] | None = None):
        with self._lock:
            self.active = {}
            self.hands = hands or []
            for g in gestures:
                if g.type not in self.bindings:
                    continue
                targets = self._targets_of(g)
                if not targets:
                    targets = [None]
                self.active[g.type] = targets
                self._ensure_assignements(g.type, len(targets))
        return self.active

    def render(self, frame: np.ndarray, renderer: Renderer) -> np.ndarray:
        with self._lock:
            for gtype, targets in self.active.items():
                effects = self.assignements.get(gtype, [])
                pairs = list(zip(targets, effects, strict=True))
                pairs.sort(key=lambda p: getattr(p[0], "depth", 0.0), reverse=True)
                for target, effect in pairs:
                    frame = effect.apply(frame, target, renderer, hands=self.hands)
        return frame

    def snapshot(self) -> dict[GestureType, dict[str, tuple[type, any]]]:
        with self._lock:
            return {
                gtype.value: {
                    "regions": len(regions),
                    "effects": [e.name for e in self.assignements.get(gtype, [])],
                }
                for gtype, regions in self.active.items()
            }

    def set_pool(self, gtype: GestureType, effects: list[Effect]):
        with self._lock:
            b = dict(self.bindings)
            if effects:
                b[gtype] = effects
            else:
                b.pop(gtype, None)
            self.bindings = b
            self.assignements.pop(gtype, None)

    def clear(self):
        with self._lock:
            self.assignements.clear()
            self.active.clear()

    def _regions_of(self, g: DetectedGesture) -> list[NormalizedBox | NormalizedQuad]:
        if g.regions:
            return g.regions
        return [g.region] if g.region is not None else []

    def _targets_of(self, g: DetectedGesture) -> list[RenderTarget]:
        if g.regions:
            return [RenderTarget(region=region) for region in g.regions]
        if g.region:
            return [RenderTarget(region=g.region)]
        if g.points:
            return [RenderTarget(point=point) for point in g.points]
        return []

    def _ensure_assignements(self, gtype: GestureType, n: int):
        pool = self.bindings.get(gtype, [])
        if not pool:
            return
        existing = self.assignements.get(gtype, [])
        if existing is not None and len(existing) == n:
            return
        self.assignements[gtype] = [pool[i % len(pool)] for i in range(n)]
