from __future__ import annotations

import argparse
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import List

import pandas as pd

from systems.block_planner import parse_duration, plan_blocks
from systems.data_loader import load_or_fetch


logger = logging.getLogger("bot")


def main(argv: List[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", default="regimes")
    parser.add_argument("--tag", required=True, help="Asset tag to load")
    parser.add_argument("--train", required=True, help="Training window (e.g. 3m)")
    parser.add_argument("--test", required=True, help="Testing window (e.g. 1m)")
    parser.add_argument("--step", required=True, help="Step size between blocks")
    parser.add_argument("--fetch-all", action="store_true", help="Fetch full history from Binance")
    parser.add_argument("--start", help="Range start for Kraken fetch")
    parser.add_argument("--end", help="Range end for Kraken fetch")
    parser.add_argument("-v", action="count", default=0, dest="verbosity")
    args = parser.parse_args(argv)

    logging.basicConfig(level=max(logging.WARNING - args.verbosity * 10, logging.DEBUG))

    if args.mode != "regimes":
        logger.error("Unsupported mode: %s", args.mode)
        return

    df = load_or_fetch(args.tag, fetch_all=args.fetch_all, start=args.start, end=args.end)
    if df.empty:
        logger.warning("No candles loaded")
        return
    first = pd.to_datetime(df.loc[0, "timestamp"], unit="s", utc=True)
    last = pd.to_datetime(df.loc[len(df) - 1, "timestamp"], unit="s", utc=True)
    logger.info("Loaded %d candles from %s to %s", len(df), first, last)

    train_len = parse_duration(args.train)
    test_len = parse_duration(args.test)
    step_len = parse_duration(args.step)

    blocks = plan_blocks(df, train_len, test_len, step_len)

    rows = []
    for idx, b in enumerate(blocks, start=1):
        row = {
            "Block": idx,
            "Train Start": pd.to_datetime(b["train_start"], unit="s"),
            "Train End": pd.to_datetime(b["train_end"], unit="s"),
            "Test Start": pd.to_datetime(b["test_start"], unit="s"),
            "Test End": pd.to_datetime(b["test_end"], unit="s"),
            "Train Candles": b["train_candles"],
            "Test Candles": b["test_candles"],
        }
        if args.verbosity >= 2:
            row["Train Idx"] = f"{b['train_index_start']}-{b['train_index_end']}"
            row["Test Idx"] = f"{b['test_index_start']}-{b['test_index_end']}"
        rows.append(row)

    table = pd.DataFrame(rows)
    if not table.empty:
        print(table.to_string(index=False))
    else:
        logger.warning("No blocks generated")

    # Save plan
    logs_dir = Path("logs")
    logs_dir.mkdir(exist_ok=True)
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    out_path = logs_dir / f"block_plan_{args.tag}_{timestamp}.json"
    with out_path.open("w") as fh:
        json.dump(blocks, fh, indent=2)
    logger.info("Saved block plan to %s", out_path)


if __name__ == "__main__":
    main()
