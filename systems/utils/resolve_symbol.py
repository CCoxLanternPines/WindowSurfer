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
        "binance_tag",
    ]
    missing = [field for field in required if not meta.get(field)]
    if missing:
        raise RuntimeError(
            f"Snapshot metadata for tag '{tag}' missing fields: {', '.join(missing)}"
        )

    resolved = {
        "tag": tag,
        "fiat": fiat,
        **{field: meta[field] for field in required},
        "window_settings": base_cfg.get("window_settings", {}),
    }

    return resolved


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
    return {
        "kraken": ledger.get("kraken_name"),
        "binance": ledger.get("binance_tag"),
    }
