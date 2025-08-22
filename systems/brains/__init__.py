from .exhaustion import ExhaustionBrain

REGISTRY = {
    "exhaustion": ExhaustionBrain,
}

def load_brain(name: str):
    cls = REGISTRY.get(name)
    if not cls:
        raise ValueError(f"Unknown brain: {name}")
    return cls()
