"""Helpers for resolving symbol and ledger configuration."""

from __future__ import annotations

import json
from pathlib import Path

from systems.utils.settings_loader import load_settings


SETTINGS = load_settings()

SNAPSHOT_DIR = Path("data/snapshots")
KRAKEN_FILE = SNAPSHOT_DIR / "kraken_symbols.json"

# Static fallback mapping for Binance symbols if not specified in settings
BINANCE_FALLBACKS = {
    "DOGEUSD": "DOGEUSDT",
    "SOLUSD": "SOLUSDT",
}


def resolve_ledger_settings(ledger_name: str, settings: dict | None = None) -> dict:
    """Return resolved ledger configuration for ``ledger_name``.

    The base configuration contains only ``tag``, ``fiat`` and
    ``window_settings``. All exchange-specific metadata is resolved at runtime
    using cached symbol snapshots. A ``RuntimeError`` is raised if any required
    metadata field is missing.
    """
    cfg = settings or SETTINGS
    try:
        base_cfg = cfg.get("ledger_settings", {})[ledger_name]
    except KeyError as exc:
        raise ValueError(f"No ledger found for name: {ledger_name}") from exc

    tag = base_cfg.get("tag")
    fiat = base_cfg.get("fiat")
    if not tag or not fiat:
        raise ValueError(f"Ledger '{ledger_name}' missing 'tag' or 'fiat'")

    meta = resolve_symbol_metadata(tag)
    required = [
        "kraken_tag",
        "kraken_pair",
        "kraken_name",
        "wallet_code",
        "fiat_code",
    ]
    missing = [field for field in required if not meta.get(field)]
    if missing:
        raise RuntimeError(
            f"Snapshot metadata for tag '{tag}' missing fields: {', '.join(missing)}"
        )

    binance_name = base_cfg.get("binance_name") or BINANCE_FALLBACKS.get(tag)

    resolved = {
        "tag": tag,
        "fiat": fiat,
        **{field: meta[field] for field in required},
        "binance_name": binance_name,
        "window_settings": base_cfg.get("window_settings", {}),
    }

    return resolved


def resolve_symbol_metadata(tag: str) -> dict:
    """Return cached Kraken symbol metadata for ``tag``."""
    try:
        kraken_map = json.loads(KRAKEN_FILE.read_text())
    except FileNotFoundError:
        kraken_map = {}

    meta = kraken_map.get(tag)
    if not meta:
        raise ValueError(f"No Kraken symbol metadata found for tag: {tag}")

    return meta


def resolve_symbol(ledger_name: str) -> dict:
    """Resolve exchange-specific pair names for ``ledger_name``."""
    ledger = resolve_ledger_settings(ledger_name)
    return {
        "kraken": ledger.get("kraken_name"),
        "binance": ledger.get("binance_name"),
    }
