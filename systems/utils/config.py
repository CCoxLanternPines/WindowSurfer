from __future__ import annotations

"""Configuration loading and resolution helpers."""

from pathlib import Path
import json
import yaml
from typing import Any, Dict
import copy

ROOT = Path(__file__).resolve().parents[2]

DEFAULT_CFG: Dict[str, Any] = {
    "window_alg": {"window": 300, "step": 5},
    "base_unit": 1.0,
    "topbottom": {
        "alpha_wick": 0.12,
        "smooth_ema": 0.25,
        "momentum_bars": 8,
        "momentum_eps": 0.0015,
        "dead_zone_min": 0.44,
        "dead_zone_max": 0.56,
        "dead_zone_pct": None,
    },
    "snapback_odds": {
        "lookback": 8,
        "weights": {
            "divergence": 0.45,
            "wick": 0.35,
            "depth": 0.20,
        },
    },
}


def _read_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def _read_yaml(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
        return {} if data is None else data


def load_ledger(path: str = "settings/ledger.json") -> Dict[str, Any]:
    """Load the project ledger configuration."""
    return _read_json(ROOT / path)


def load_global(path: str = "settings/global.yaml") -> Dict[str, Any]:
    """Load global defaults."""
    return _read_yaml(ROOT / path)


def load_knobs(path: str = "settings/knobs.json") -> Dict[str, Any]:
    """Load knob ranges for search."""
    return _read_json(ROOT / path)


def _deep_update(base: Dict[str, Any], overrides: Dict[str, Any]) -> Dict[str, Any]:
    for key, val in overrides.items():
        if isinstance(val, dict) and isinstance(base.get(key), dict):
            base[key] = _deep_update(base.get(key, {}), val)
        else:
            base[key] = val
    return base


def resolve_window_cfg(
    ledger_name: str,
    window_key: str,
    tag: str,
    *,
    ledger: Dict[str, Any],
    global_cfg: Dict[str, Any],
) -> Dict[str, Any]:
    """Resolve configuration for a specific ledger/window pair."""

    ledgers = ledger.get("ledger_settings")
    if not ledgers:
        raise ValueError("ledger_settings missing in ledger config")
    if ledger_name not in ledgers:
        raise ValueError(f"ledger '{ledger_name}' not found")
    ledger_cfg = ledgers[ledger_name]
    windows = ledger_cfg.get("window_settings") or {}
    if window_key not in windows:
        raise ValueError(
            f"window '{window_key}' not found for ledger '{ledger_name}'"
        )
    window_cfg = windows[window_key]

    cfg = copy.deepcopy(DEFAULT_CFG)

    # Global defaults
    _deep_update(cfg, global_cfg.get("engine_defaults", {}))
    strat = global_cfg.get("strategy_defaults", {})
    if strat.get("topbottom"):
        _deep_update(cfg["topbottom"], strat["topbottom"])
    weights = (strat.get("snapback_odds") or {}).get("weights")
    if weights:
        _deep_update(cfg["snapback_odds"]["weights"], weights)
    if (strat.get("snapback_odds") or {}).get("lookback") is not None:
        cfg["snapback_odds"]["lookback"] = strat["snapback_odds"]["lookback"]
    if strat.get("odds_lookback") is not None:
        cfg["snapback_odds"]["lookback"] = strat["odds_lookback"]

    # Ledger level overrides
    ledger_overlay = {
        k: ledger_cfg[k]
        for k in ("window_alg", "base_unit", "topbottom", "snapback_odds")
        if k in ledger_cfg
    }
    _deep_update(cfg, ledger_overlay)

    # Window level overrides
    win_overlay = {
        k: window_cfg[k]
        for k in ("window_alg", "base_unit", "topbottom", "snapback_odds")
        if k in window_cfg
    }
    _deep_update(cfg, win_overlay)

    # Legacy support for dead_zone_pct at root
    if "dead_zone_pct" in window_cfg:
        cfg["topbottom"]["dead_zone_pct"] = window_cfg["dead_zone_pct"]

    return cfg
