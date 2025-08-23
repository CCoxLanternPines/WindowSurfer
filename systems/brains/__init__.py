from .exhaustion import ExhaustionBrain
from .velocity import VelocityBrain

REGISTRY = {
    "exhaustion": ExhaustionBrain,
    "velocity": VelocityBrain,
}

def list_brains() -> list[str]:
    return sorted(REGISTRY.keys())

def load_brain(name: str):
    if name not in REGISTRY:
        raise ValueError(f"Unknown brain: {name}")
    return REGISTRY[name]()
