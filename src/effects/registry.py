from src.effects.effect import Effect

EFFECT_REGISTRY: dict[str, type[Effect]] = {}


def register_effect(name: str):
    def deco(cls):
        EFFECT_REGISTRY[name] = cls
        return cls

    return deco
