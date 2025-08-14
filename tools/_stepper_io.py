"""CSV loading and path resolution helpers for candle stepper."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd

# Base directories relative to repo root
REPO_ROOT = Path(__file__).resolve().parents[1]
SIM_DIR = REPO_ROOT / "data" / "sim"
RAW_DIR = REPO_ROOT / "data" / "raw"


def resolve_csv(tag: Optional[str] = None, csv: Optional[str] = None) -> Path:
    """Resolve a CSV path from a tag or explicit path.

    Precedence: explicit ``csv`` argument > ``SIM_DIR`` > ``RAW_DIR``.
    ``tag`` should not include an extension.  When ``csv`` is provided the
    path is interpreted relative to the repository root.
    """

    attempted: list[Path] = []

    if csv:
        path = (REPO_ROOT / csv.lstrip("/")).resolve()
        if path.exists():
            return path
        attempted.append(path)
        raise FileNotFoundError(
            f"CSV not found at {path}. Provide a valid path with --csv.")

    if tag:
        path = (SIM_DIR / f"{tag}.csv").resolve()
        if path.exists():
            return path
        attempted.append(path)

        path = (RAW_DIR / f"{tag}.csv").resolve()
        if path.exists():
            return path
        attempted.append(path)

    attempted_str = "\n".join(str(p) for p in attempted)
    raise FileNotFoundError(
        f"Could not locate CSV for tag '{tag}'. Tried:\n{attempted_str}\n"
        "Use --csv <path> to specify a file explicitly.")


def read_candles(csv_path: Path) -> pd.DataFrame:
    """Read and normalize candle data from ``csv_path``.

    Columns are lower-cased with timestamp synonyms normalised.
    Required columns: timestamp, open, high, low, close.  Volume is optional.
    Timestamps are parsed in UTC, invalid rows dropped, sorted and reindexed.
    """

    df = pd.read_csv(
        csv_path,
        engine="python",
        on_bad_lines="skip",
        skip_blank_lines=True,
    )

    df.columns = [c.strip().lower() for c in df.columns]
    rename_map = {c: "timestamp" for c in ["timestamp", "time", "date"] if c in df.columns}
    df = df.rename(columns=rename_map)

    required = {"timestamp", "open", "high", "low", "close"}
    missing = required - set(df.columns)
    if missing:
        sample_cols = ", ".join(list(df.columns)[:3])
        raise ValueError(
            f"Missing required columns: {missing}. First columns: {sample_cols}")

    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    df = df.dropna(subset=["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)

    return df
