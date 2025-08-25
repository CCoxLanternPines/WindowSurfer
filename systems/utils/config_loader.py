from __future__ import annotations

"""Load application settings with caching."""

import json
from pathlib import Path
from typing import Any, Dict

_SETTINGS_CACHE: Dict[str, Any] | None = None


def load_settings(*, reload: bool = False) -> Dict[str, Any]:
    """Return the contents of settings/settings.json."""
    global _SETTINGS_CACHE
    if _SETTINGS_CACHE is None or reload:
        path = Path(__file__).resolve().parents[2] / "settings" / "settings.json"
        with path.open("r", encoding="utf-8") as fh:
            _SETTINGS_CACHE = json.load(fh)
    return _SETTINGS_CACHE


def get_viz_filters() -> Dict[str, Any]:
    """Return visualization filter settings with safe defaults."""
    data = load_settings()
    vf = data.get("viz_filters", {})
    try:
        vol = float(vf.get("volatility_min_size", 0) or 0)
    except (TypeError, ValueError):
        vol = 0.0
    try:
        pres = float(vf.get("pressure_min_size", 0) or 0)
    except (TypeError, ValueError):
        pres = 0.0
    try:
        ang = int(vf.get("angle_skip_n", 1) or 1)
    except (TypeError, ValueError):
        ang = 1
    return {
        "volatility_min_size": max(0.0, vol),
        "pressure_min_size": max(0.0, pres),
        "angle_skip_n": max(1, ang),
    }
