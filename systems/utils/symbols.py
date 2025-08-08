from __future__ import annotations

"""Canonical symbol helpers for asset/tag resolution and paths."""

from pathlib import Path
from typing import Dict, Any

from systems.utils.config import resolve_path

_FIAT_SUFFIXES = ("USD", "USDT", "USDC", "DAI")


def resolve_asset(ledger_cfg: Dict[str, Any]) -> str:
    """Return the base asset symbol for ``ledger_cfg``.

    Priority:
    1. ``ledger_cfg['asset']`` if provided.
    2. Strip a terminal fiat suffix from ``ledger_cfg['tag']``.
    3. Base portion of ``kraken_name``/``binance_name`` (split on '/').
    """

    asset = ledger_cfg.get("asset")
    if asset:
        return asset.upper()

    tag = ledger_cfg.get("tag", "")
    if tag:
        tag = tag.upper()
        for suffix in _FIAT_SUFFIXES:
            if tag.endswith(suffix):
                return tag[: -len(suffix)]
        return tag

    pair = ledger_cfg.get("kraken_name") or ledger_cfg.get("binance_name") or ""
    if pair:
        base = pair.split("/")[0]
        return base.upper()
    raise ValueError("Could not resolve asset from ledger config")


def resolve_tag(ledger_cfg: Dict[str, Any]) -> str:
    """Return the full pair tag for ``ledger_cfg``."""

    tag = ledger_cfg.get("tag")
    if tag:
        return tag.upper()
    asset = resolve_asset(ledger_cfg)
    fiat = ledger_cfg.get("preferred_fiat", "USD")
    return f"{asset}{fiat.upper()}"


def resolve_exchange_pairs(ledger_cfg: Dict[str, Any]) -> Dict[str, str | None]:
    """Return exchange pair mappings without guessing."""

    return {
        "kraken_pair": ledger_cfg.get("kraken_pair"),
        "binance_pair": ledger_cfg.get("binance_name"),
    }


def raw_path(asset: str) -> Path:
    """Return path to the raw candle CSV for ``asset``."""

    root = resolve_path("")
    return root / "data" / "raw" / f"{asset.upper()}.csv"


def live_ledger_path(asset: str) -> Path:
    """Return path to the live ledger JSON for ``asset``."""

    root = resolve_path("")
    return root / "data" / "ledgers" / f"{asset.upper()}.json"


def sim_ledger_path(asset: str) -> Path:
    """Return path to the simulation ledger JSON for ``asset``."""

    root = resolve_path("")
    return root / "data" / "tmp" / "simulation" / f"{asset.upper()}.json"


def resolve_knobs(ledger_cfg: Dict[str, Any], knobs: Dict[str, Any]) -> Dict[str, Any] | None:
    """Return knob configuration for ``ledger_cfg``.

    Looks for ``knobs`` first under the tag key, then under the asset key.
    """

    tag = ledger_cfg.get("tag", "").upper()
    asset = resolve_asset(ledger_cfg)
    return knobs.get(tag) or knobs.get(asset)


__all__ = [
    "resolve_asset",
    "resolve_tag",
    "resolve_exchange_pairs",
    "raw_path",
    "live_ledger_path",
    "sim_ledger_path",
    "resolve_knobs",
]
