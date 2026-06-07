import json
from dataclasses import dataclass, field

from loguru import logger

from src.effects.effect import current_params
from src.effects.effect_manager import EffectManager
from src.gestures.gesture_manager import DetectedGesture
from src.gestures.registry import GESTURE_REGISTRY


@dataclass
class BindingConfig:
    gesture: str
    effect: str | None = None
    effects: list[str] | None = None
    effect_params: dict = field(default_factory=dict)

    def __post_init__(self):
        raw = self.effects if self.effects is not None else [self.effect]
        spec: list[tuple[str, dict[str, any]]] = []
        for item in raw:
            logger.info(f"Processing item: {item}")
            if isinstance(item, str):
                spec.append((item, dict(self.effect_params.get(item, {}))))
            else:
                spec.append((item["name"], dict(item.get("params", {}))))
        self.effect_params = dict(spec)


@dataclass
class AppConfig:
    bindings: list[BindingConfig]
    gesture_params: dict = field(default_factory=dict)

    @staticmethod
    def from_json(json_path: str) -> "AppConfig":
        with open(json_path) as f:
            data = json.load(f)
        return AppConfig(
            bindings=[BindingConfig(**binding) for binding in data["bindings"]],
            gesture_params={
                k: v for k, v in data["gesture_params"].items() if k in GESTURE_REGISTRY
            },
        )


def save_config(
    path: str, effect_manager: EffectManager, detectors_by_name: dict[str, DetectedGesture]
) -> None:
    bindings = []
    for gtype, pool in effect_manager.bindings.items():
        bindings.append(
            {
                "gesture": gtype.value,
                "effects": [{"name": e.name, "params": current_params(e)} for e in pool],
            }
        )
    gesture_params = {
        name: current_params(detector)
        for name, detector in detectors_by_name.items()
        if name in GESTURE_REGISTRY
    }
    with open(path, "w") as f:
        json.dump({"bindings": bindings, "gesture_params": gesture_params}, f, indent=4)
