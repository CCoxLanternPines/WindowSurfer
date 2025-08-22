from typing import Dict, Type
from .base import Brain
from .exhaustion import ExhaustionBrain

_REGISTRY: Dict[str, Type[Brain]] = {
    "exhaustion": ExhaustionBrain,
    # add more brains here later: "vol_compress": VolCompressionBrain, ...
}


def get_brain(name: str) -> Brain:
    if name not in _REGISTRY:
        raise ValueError(f"Unknown brain '{name}'. Available: {list(_REGISTRY)}")
    return _REGISTRY[name]()  # construct
