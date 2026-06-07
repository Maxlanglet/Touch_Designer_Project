from loguru import logger

from src.config.config import AppConfig
from src.effects.effect_manager import EffectManager
from src.effects.registry import EFFECT_REGISTRY
from src.gestures.gesture_manager import GestureManager, GestureType
from src.gestures.registry import GESTURE_REGISTRY


def build_from_config(config: AppConfig) -> tuple[GestureManager, EffectManager]:
    logger.info(f"Building gesture manager from config: {config}")
    logger.info(f"Gesture params: {config.gesture_params}")
    logger.info(f"Bindings: {config.bindings}")
    logger.info(f"Gesture registry: {GESTURE_REGISTRY}")
    logger.info(f"Effect registry: {EFFECT_REGISTRY}")
    detectors = [
        GESTURE_REGISTRY[binding.gesture](**config.gesture_params.get(binding.gesture, {}))
        for binding in config.bindings
    ]
    bindings: dict[GestureType, list] = {}
    for b in config.bindings:
        pool = [EFFECT_REGISTRY[name](**params) for name, params in b.effect_params.items()]
        bindings[GestureType(b.gesture)] = pool
    return GestureManager(detectors=detectors), EffectManager(bindings=bindings)
