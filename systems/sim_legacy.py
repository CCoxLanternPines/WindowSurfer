from __future__ import annotations

"""Legacy simulation loader with strict D/W/M/Y window filtering.

This module loads CSV candle data for a given symbol and optionally
applies a lookback window relative to the dataset's maximum timestamp
in the file. The ``--time`` parameter is a window filter, not a file
selector—filenames are never altered by ``--time``.

Accepted units: D, W, M, Y (uppercase only).
Fixed durations are used for parity:

``D=1 day, W=7 days, M=30 days, Y=365 days.``
"""

from datetime import datetime, timedelta, timezone
import re
from pathlib import Path

import pandas as pd

DATA_DIR = Path("data")
CANDLES_DIR = DATA_DIR / "candles" / "sim"

_TIME_RE = re.compile(r"^[1-9]\d*[DWMY]$")
_UNIT_DAYS = {"D": 1, "W": 7, "M": 30, "Y": 365}

# Try to leverage legacy time utils if present; fall back to strict parser.
_LEGACY_TIME_FN = None
try:
    from util import time as util_time  # type: ignore

    for _cand in (
        "parse_lookback_to_seconds",
        "window_to_seconds",
        "parse_window",
    ):
        if hasattr(util_time, _cand):
            _LEGACY_TIME_FN = getattr(util_time, _cand)
            break
except Exception:  # pragma: no cover - optional dependency
    _LEGACY_TIME_FN = None


def _format_market(symbol: str) -> str:
    """Return a display-friendly market string like ``DOGE/USD``."""
    s = symbol.upper()
    if s.endswith("USDC") or s.endswith("USDT"):
        return f"{s[:-4]}/{s[-4:]}"
    return f"{s[:-3]}/{s[-3:]}"


def _parse_window(value: str | None) -> timedelta | None:
    """Parse ``--time`` values like ``1W`` or ``3M`` into ``timedelta``.

    Strictly enforces D/W/M/Y uppercase units.

    If ``util/time.py`` exposes a compatible helper, use it; otherwise use
    fixed-day rules.
    """
    if value is None:
        return None
    if not _TIME_RE.fullmatch(value):
        print(
            f"[ERROR] Bad --time value: \"{value}\" (expected e.g. 7D | 2W | 3M | 1Y)"
        )
        raise SystemExit(1)

    # Prefer legacy helper if present, coercing to timedelta.
    if _LEGACY_TIME_FN is not None:
        try:
            out = _LEGACY_TIME_FN(value)
            if isinstance(out, timedelta):
                return out
            seconds = int(out)
            if seconds <= 0:
                raise ValueError("non-positive seconds from legacy parser")
            return timedelta(seconds=seconds)
        except Exception:
            pass

    num = int(value[:-1])
    unit = value[-1]
    days = num * _UNIT_DAYS[unit]
    return timedelta(days=days)


def load(symbol: str, window: str | None = None) -> pd.DataFrame:
    """Load candle data for ``symbol`` with optional lookback ``window``.

    Parameters
    ----------
    symbol:
        Market pair such as ``DOGEUSD``.
    window:
        Lookback window string like ``1W`` or ``3M``. If omitted, the full
        dataset is returned.
    """

    path = CANDLES_DIR / f"{symbol}.csv"
    abs_path = path.resolve()
    if not path.exists():
        print(f"[ERROR] Missing candles: {abs_path}")
        raise SystemExit(1)

    df = pd.read_csv(path)
    rows_in = len(df)

    if "timestamp" not in df.columns:
        raise ValueError("CSV missing 'timestamp' column")

    # Support integer epoch seconds or ISO8601 strings.
    if pd.api.types.is_numeric_dtype(df["timestamp"]):
        df["timestamp"] = pd.to_numeric(df["timestamp"], errors="coerce")
        df = df.dropna(subset=["timestamp"]).sort_values("timestamp")
        df["timestamp"] = df["timestamp"].astype("int64")
        t_max = int(df["timestamp"].max()) if rows_in else 0
        t_max_dt = datetime.fromtimestamp(t_max, tz=timezone.utc)
        cutoff_ts = t_max
    else:
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
        df = df.dropna(subset=["timestamp"]).sort_values("timestamp")
        t_max_dt = pd.to_datetime(df["timestamp"].max(), utc=True)
        t_max = int(t_max_dt.timestamp()) if rows_in else 0
        cutoff_ts = t_max

    delta = _parse_window(window)
    if delta is not None:
        cutoff_ts = t_max - int(delta.total_seconds())
        # Filter respecting whichever representation we normalized to above:
        if pd.api.types.is_numeric_dtype(df["timestamp"]):
            df = df[df["timestamp"] >= cutoff_ts]
        else:
            cutoff_dt = datetime.fromtimestamp(cutoff_ts, tz=timezone.utc)
            df = df[df["timestamp"] >= pd.Timestamp(cutoff_dt)]

    rows_out = len(df)
    cutoff_dt = datetime.fromtimestamp(cutoff_ts, tz=timezone.utc)
    market = _format_market(symbol)
    window_str = window if window is not None else "full"
    print(
        "[SIM][LEGACY] market="
        f"{market} window={window_str} rows_in={rows_in} rows_out={rows_out} "
        f"t_max={t_max_dt.strftime('%Y-%m-%dT%H:%M:%S')}Z "
        f"cutoff={cutoff_dt.strftime('%Y-%m-%dT%H:%M:%S')}Z"
    )
    return df.reset_index(drop=True)


def run_legacy_sim(*args, **kwargs):
    """Backward-compat shim to satisfy older imports.

    Historically, ``bot.py`` imported ``run_legacy_sim`` from this module.

    For loader-only behavior, alias to :func:`load` while accepting flexible
    arguments.

    Usage examples we support::

        run_legacy_sim("DOGEUSD", "1W")
        run_legacy_sim(symbol="DOGEUSD", window="1W")
    """

    symbol = None
    window = None

    if args:
        symbol = args[0]
        if len(args) > 1:
            window = args[1]

    symbol = kwargs.get("symbol", symbol)
    window = kwargs.get("window", window)

    if symbol is None:
        print("[ERROR] run_legacy_sim requires a symbol like 'DOGEUSD'")
        raise SystemExit(1)

    return load(symbol=symbol, window=window)


__all__ = ["load", "run_legacy_sim"]
