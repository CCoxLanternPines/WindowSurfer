from __future__ import annotations

"""Configuration helper utilities."""

from pathlib import Path
import json
from typing import Any, Dict, List

from systems.utils.addlog import addlog

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_SETTINGS_CACHE: Dict[str, Any] | None = None
_DEPRECATION_WARNED = False

_DEPRECATED_KEYS = {
    "buy_cooldown",
    "sell_cooldown",
    "maturity_multiplier",
    "buy_multiplier_scale",
    "buy_cooldown_multiplier_scale",
    "sell_cooldown_multiplier_scale",
    "dead_zone_pct",
    "buy_floor",
    "sell_ceiling",
    "cooldown",
}


def _warn_deprecated(settings: Dict[str, Any]) -> None:
    global _DEPRECATION_WARNED
    if _DEPRECATION_WARNED:
        return
    found = set()
    for ledger in settings.get("ledger_settings", {}).values():
        for win in ledger.get("window_settings", {}).values():
            found.update(key for key in win if key in _DEPRECATED_KEYS)
    if found:
        addlog(
            f"[WARN] Deprecated config keys detected: {', '.join(sorted(found))}",
            verbose_int=1,
            verbose_state=True,
        )
        _DEPRECATION_WARNED = True


def resolve_path(rel_path: str) -> Path:
    """Return an absolute path for ``rel_path`` from the project root."""
    return _PROJECT_ROOT / rel_path


def load_settings(*, reload: bool = False) -> Dict[str, Any]:
    """Load settings from ``settings/settings.json`` with optional caching."""
    global _SETTINGS_CACHE
    if _SETTINGS_CACHE is None or reload:
        settings_path = resolve_path("settings/settings.json")
        with settings_path.open("r", encoding="utf-8") as fh:
            raw = fh.read()

        dup_flag = False

        def _convert(obj: Any, path: List[str]) -> Any:
            nonlocal dup_flag
            if isinstance(obj, list):
                d: Dict[str, Any] = {}
                seen: List[str] = []
                for k, v in obj:
                    if k in d and path and path[-1] == "window_settings":
                        dup_flag = True
                    seen.append(k)
                    d[k] = _convert(v, path + [k])
                return d
            return obj

        parsed = json.loads(raw, object_pairs_hook=list)
        _SETTINGS_CACHE = _convert(parsed, [])
        _warn_deprecated(_SETTINGS_CACHE)
        if dup_flag:
            addlog(
                "[WARN] window_settings contains duplicate keys; only the last occurrence is kept by JSON. Ensure unique names (e.g., minnow, fish, dolphin, whale).",
                verbose_int=1,
                verbose_state=True,
            )
    return _SETTINGS_CACHE


def load_ledger_config(ledger_name: str) -> Dict[str, Any]:
    """Return the configuration dictionary for a given ledger.

    Parameters
    ----------
    ledger_name: str
        Name of the ledger to load from settings.

    Raises
    ------
    ValueError
        If the requested ledger does not exist in ``settings.json``.
    """

    settings = load_settings()
    ledgers = settings.get("ledger_settings", {})
    if ledger_name not in ledgers:
        raise ValueError(f"Ledger '{ledger_name}' not found in settings")

    ledger_cfg: Dict[str, Any] = dict(ledgers[ledger_name])

    # ------------------------------------------------------------------
    # Canonicalise coin/fiat settings and drop legacy tag usage.
    # ------------------------------------------------------------------
    def _split_legacy_tag(t: str) -> tuple[str, str]:
        t = t.strip().upper()
        for q in ("USDT", "USDC", "USD", "EUR", "GBP", "BTC", "ETH"):
            if t.endswith(q):
                return t[:-len(q)], q
        raise ValueError(f"Cannot derive coin/fiat from legacy tag '{t}'")

    coin = ledger_cfg.get("coin")
    fiat = ledger_cfg.get("fiat")
    if (not coin or not fiat) and ledger_cfg.get("tag"):
        coin, fiat = _split_legacy_tag(str(ledger_cfg["tag"]))
    if not isinstance(coin, str) or not coin.strip():
        raise ValueError(f"Ledger '{ledger_name}' missing 'coin'")
    if not isinstance(fiat, str) or not fiat.strip():
        raise ValueError(f"Ledger '{ledger_name}' missing 'fiat'")
    ledger_cfg["coin"] = coin.strip().upper()
    ledger_cfg["fiat"] = fiat.strip().upper()
    ledger_cfg.pop("tag", None)

    # Warn if multiple ledgers share the same coin with different fiats.
    _coin_seen = getattr(load_ledger_config, "_coin_seen", {})
    prev = _coin_seen.get(ledger_cfg["coin"])
    if prev and prev != ledger_cfg["fiat"]:
        addlog(
            f"[WARN] Multiple ledgers share COIN={ledger_cfg['coin']}; raw file is shared.",
            verbose_int=1,
            verbose_state=True,
        )
    else:
        _coin_seen[ledger_cfg["coin"]] = ledger_cfg["fiat"]
    setattr(load_ledger_config, "_coin_seen", _coin_seen)

    addlog(
        f"[ID] Using market {ledger_cfg['coin']}/{ledger_cfg['fiat']}; raw file = data/raw/{ledger_cfg['coin'].upper()}.csv",
        verbose_int=1,
        verbose_state=True,
    )

    return ledger_cfg
