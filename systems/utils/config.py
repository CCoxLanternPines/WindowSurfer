from __future__ import annotations

"""Configuration helper utilities."""

from pathlib import Path
import json
from typing import Any, Dict, List

from systems.utils.addlog import addlog
from systems.utils.resolve_symbol import resolve_symbols, to_tag

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
        for name, ledger in _SETTINGS_CACHE.get("ledger_settings", {}).items():
            for key in ("tag", "wallet_code", "kraken_pair", "binance_name"):
                if key in ledger:
                    addlog(
                        f"[DEPRECATED] ledger '{name}' field '{key}' is ignored; use kraken_name only",
                        verbose_int=1,
                        verbose_state=True,
                    )
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
    return ledgers[ledger_name]


def resolve_ccxt_symbols_by_coin(coin: str) -> tuple[str, str]:
    """Return Kraken and Binance symbols for ``coin``.

    The first ledger whose base starts with ``coin`` (case-insensitive) is used.
    Logs which ledger tag was matched.
    """

    settings = load_settings()
    coin_up = coin.upper()
    for cfg in settings.get("ledger_settings", {}).values():
        kraken_name = cfg.get("kraken_name", "")
        if not kraken_name:
            continue
        symbols = resolve_symbols(kraken_name)
        tag = to_tag(symbols["kraken_name"])
        base = symbols["kraken_name"].split("/")[0].upper()
        if base.startswith(coin_up):
            addlog(
                f"[CONFIG] coin={coin_up} resolved using tag={tag}",
                verbose_int=1,
                verbose_state=True,
            )
            return symbols["kraken_name"], symbols["binance_name"]
    msg = (
        f"[ERROR] No ledger maps coin={coin_up} to exchange symbols. "
        f"Add a ledger with kraken_name for the coin."
    )
    addlog(msg, verbose_int=1, verbose_state=True)
    raise ValueError(msg)
