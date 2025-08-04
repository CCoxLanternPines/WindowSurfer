"""Helpers for resolving symbol and ledger configuration."""

from __future__ import annotations

import json
from pathlib import Path

from systems.utils.settings_loader import load_settings


SETTINGS = load_settings()

SNAPSHOT_DIR = Path("data/snapshots")
KRAKEN_FILE = SNAPSHOT_DIR / "kraken_symbols.json"
BINANCE_FILE = SNAPSHOT_DIR / "binance_symbols.json"


def resolve_ledger_settings(ledger_name: str, settings: dict | None = None) -> dict:
    """Return ledger configuration for ``ledger_name``."""
    cfg = settings or SETTINGS
    try:
        return cfg.get("ledger_settings", {})[ledger_name]
    except KeyError:
        raise ValueError(f"No ledger found for name: {ledger_name}") from None


def resolve_symbol_metadata(tag: str) -> dict:
    """Return cached symbol metadata for ``tag`` from both exchanges."""
    try:
        kraken_map = json.loads(KRAKEN_FILE.read_text())
    except FileNotFoundError:
        kraken_map = {}

    try:
        binance_map = json.loads(BINANCE_FILE.read_text())
    except FileNotFoundError:
        binance_map = {}

    meta: dict[str, str] = {}
    if tag in kraken_map:
        meta.update(kraken_map[tag])
    if tag in binance_map:
        meta.update(binance_map[tag])

    if not meta:
        raise ValueError(f"No symbol metadata found for tag: {tag}")

    return meta


def resolve_symbol(ledger_name: str) -> dict:
    """Resolve exchange-specific pair names for ``ledger_name``."""
    ledger = resolve_ledger_settings(ledger_name)
    tag = ledger.get("tag")
    if not tag:
        raise ValueError(f"Ledger '{ledger_name}' missing required 'tag'")
    meta = resolve_symbol_metadata(tag)
    return {
        "kraken": meta.get("kraken_name"),
        "binance": meta.get("binance_tag"),
    }
