from __future__ import annotations

"""Shared candle loading utilities for sim and live engines."""

from pathlib import Path
from typing import Tuple

import pandas as pd

from systems.utils.addlog import addlog
from systems.utils.resolve_symbol import (
    candle_filename,
    sim_path_csv,
    live_path_csv,
    to_tag,
)


def load_candles_df(
    account: str,
    market: str,
    *,
    live: bool = False,
    verbose: int = 0,
) -> Tuple[pd.DataFrame, int]:
    """Return normalised candles for ``account``/``market``.

    Parameters
    ----------
    account: str
        Account name used for the candle filename.
    market: str
        Market pair in CCXT format, e.g. ``"SOL/USD"``.
    live: bool, optional
        If ``True``, load from ``data/live`` else ``data/sim``.
    verbose: int, optional
        Verbosity level forwarded to ``addlog``.

    Returns
    -------
    Tuple[pd.DataFrame, int]
        Normalised dataframe and number of duplicate rows removed.
    """

    csv_path = candle_filename(account, market, live=live)
    if not Path(csv_path).exists():
        tag = to_tag(market)
        legacy = live_path_csv(tag) if live else sim_path_csv(tag)
        if Path(legacy).exists():
            addlog(
                f"[DEPRECATED] Found legacy file {legacy}, use {csv_path}",
                verbose_int=1,
                verbose_state=verbose,
            )
        raise FileNotFoundError(
            f"Missing data file: {csv_path}. Run: python bot.py --mode fetch --account {account} --market {market}"
        )

    df = pd.read_csv(csv_path)

    ts_col = next(
        (c for c in df.columns if str(c).lower() in ("timestamp", "time", "date")),
        None,
    )
    if ts_col is None:
        raise ValueError(f"No timestamp column in {csv_path}")

    df[ts_col] = pd.to_numeric(df[ts_col], errors="coerce")
    df = df.dropna(subset=[ts_col])
    before = len(df)
    df = df.sort_values(ts_col).drop_duplicates(subset=[ts_col], keep="last").reset_index(drop=True)
    removed = before - len(df)

    if ts_col != "timestamp":
        df = df.rename(columns={ts_col: "timestamp"})

    return df, removed

