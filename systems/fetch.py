from __future__ import annotations

import argparse
import pandas as pd

from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))

from systems.utils.config import load_settings
from systems.scripts.fetch_core import get_gapless_1h, FetchAbort


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Fetch gapless 1h candles")
    parser.add_argument("--tag", help="trading pair tag", required=False)
    parser.add_argument(
        "--full", action="store_true", help="force full Binance refresh"
    )
    args = parser.parse_args(argv)

    settings = load_settings()
    default_tag = settings.get("ledger_settings", {}).get("default", {}).get("tag", "")
    tag = (args.tag or default_tag).upper()

    try:
        df = get_gapless_1h(tag=tag, allow_cache=not args.full)
    except FetchAbort as e:
        print(e)
        return

    start = pd.to_datetime(df["ts"].min(), unit="ms", utc=True).strftime("%Y-%m-%d")
    end = pd.to_datetime(df["ts"].max(), unit="ms", utc=True).strftime("%Y-%m-%d")
    print(f"[FETCH] tag={tag} rows={len(df)} span={start} → {end} gaps=0")


if __name__ == "__main__":
    main()
