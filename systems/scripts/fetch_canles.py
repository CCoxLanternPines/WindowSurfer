from __future__ import annotations

"""Utilities to load historical candle data."""

import pandas as pd

from systems.utils.config import resolve_path


def _extract_coin(tag: str) -> str:
    """Return the base coin symbol from a market ``tag``.

    Tags often include a quote currency (e.g. ``SOLUSDC`` or ``DOGEUSD``).
    Candle files are stored using just the base coin name, so this function
    strips any stablecoin suffixes and returns the resulting coin symbol.
    """

    tag = tag.upper()
    suffixes = ("USDC", "USD", "DAI")
    for suffix in suffixes:
        if tag.endswith(suffix):
            return tag[: -len(suffix)]
    return tag


def fetch_candles(tag: str) -> pd.DataFrame:
    """Load historical candles for ``tag`` from ``data/raw``.

    Only the base coin portion of ``tag`` is used to determine the file name.
    """
    root = resolve_path("")
    coin = _extract_coin(tag)
    path = root / "data" / "raw" / f"{coin}.csv"
    if not path.exists():  # pragma: no cover - file presence check
        raise FileNotFoundError(f"Candle file not found: {path}")
    return pd.read_csv(path)
