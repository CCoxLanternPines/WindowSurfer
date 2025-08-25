from __future__ import annotations

"""Load application settings with caching."""

import json
from pathlib import Path
from typing import Any, Dict

_SETTINGS_CACHE: Dict[str, Any] | None = None
_COIN_SETTINGS_CACHE: Dict[str, Dict[str, Any]] = {}


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


def load_coin_settings(market: str, *, reload: bool = False) -> Dict[str, Any]:
    """Return merged coin settings for ``market``."""
    global _COIN_SETTINGS_CACHE
    if market not in _COIN_SETTINGS_CACHE or reload:
        path = Path(__file__).resolve().parents[2] / "settings" / "coin_settings.json"
        with path.open("r", encoding="utf-8") as fh:
            data = json.load(fh).get("coin_settings", {})
        cfg = dict(data.get("default", {}))
        overrides = data.get(market)
        if overrides:
            cfg.update(overrides)
        _COIN_SETTINGS_CACHE[market] = cfg
    return _COIN_SETTINGS_CACHE[market]


def get_coin_setting(coin: str, key: str, default: Any = None) -> Any:
    """Retrieve a coin-specific setting with ``default`` fallback."""
    cfg = load_coin_settings(coin)
    return cfg.get(key, default)
