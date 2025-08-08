"""Utilities to load historical candle data."""

from __future__ import annotations

from typing import Dict, Any
import pandas as pd

from systems.utils.symbols import resolve_asset, raw_path


def fetch_candles(ledger_cfg: Dict[str, Any] | None = None, *, asset: str | None = None) -> pd.DataFrame:
    """Load historical candles for an asset.

    Parameters
    ----------
    ledger_cfg: Optional configuration dictionary for the ledger.
    asset: Explicit asset symbol. Overrides ``ledger_cfg`` if provided.
    """

    if asset is None:
        if ledger_cfg is None:
            raise ValueError("ledger_cfg or asset must be provided")
        asset = resolve_asset(ledger_cfg)
    path = raw_path(asset)
    if not path.exists():  # pragma: no cover - file presence check
        raise FileNotFoundError(f"Candle file not found: {path}")
    return pd.read_csv(path)
