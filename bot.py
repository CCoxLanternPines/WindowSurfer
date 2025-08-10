from __future__ import annotations

import argparse
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import pandas as pd

from systems.block_planner import plan_blocks
from systems.cli.common_args import add_verbosity, parse_duration_1h, validate_dates
from systems.data_loader import (
    CACHE_DIR,
    fetch_all_history_binance,
    fetch_range_kraken,
    load_or_fetch,
    _load_settings,
)

logger = logging.getLogger("bot")


def resolve_ccxt_symbols(tag: str) -> tuple[str, str]:
    """Return (kraken_symbol, binance_symbol) for ``tag`` from settings.json."""
    path = Path("settings.json")
    settings = {}
    if path.exists():
        with path.open() as fh:
            settings = json.load(fh)
    info = settings.get(tag, {})
    return info.get("kraken_name", tag), info.get("binance_name", tag)


def cmd_fetch_history(args: argparse.Namespace) -> None:
    start, end = validate_dates(args.start, args.end)
    if not args.fetch_all and not (start and end):
        raise SystemExit("Provide --fetch-all or both --start and --end")

    kraken_symbol, binance_symbol = resolve_ccxt_symbols(args.tag)

    if args.fetch_all:
        df = fetch_all_history_binance(binance_symbol)
        first = pd.to_datetime(df.loc[0, "timestamp"], unit="s", utc=True)
        last = pd.to_datetime(df.loc[len(df) - 1, "timestamp"], unit="s", utc=True)
        print(
            f"[FETCH] TAG={args.tag} | Source=Binance | Candles={len(df)} | "
            f"Range={first} -> {last}"
        )
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        cache_path = CACHE_DIR / f"{args.tag}_1h.parquet"
        df.to_parquet(cache_path, index=False)
        print(f"[CACHE] {cache_path}")
    else:
        df = fetch_range_kraken(kraken_symbol, start, end)
        first = (
            pd.to_datetime(df.loc[0, "timestamp"], unit="s", utc=True)
            if not df.empty
            else None
        )
        last = (
            pd.to_datetime(df.loc[len(df) - 1, "timestamp"], unit="s", utc=True)
            if not df.empty
            else None
        )
        print(
            f"[FETCH] TAG={args.tag} | Source=Kraken | Candles={len(df)} | "
            f"Range={first} -> {last}"
        )


def cmd_regimes(args: argparse.Namespace) -> None:
    if args.cluster:
        features_dir = Path("features")
        feat_files = sorted(features_dir.glob(f"features_{args.tag}_*.parquet"))
        if not feat_files:
            raise SystemExit(f"No features found for {args.tag}; run with --features first")
        latest_feat = feat_files[-1]
        stamp = "_".join(latest_feat.stem.split("_")[2:])
        meta_path = features_dir / f"features_meta_{args.tag}_{stamp}.json"
        if not meta_path.exists():
            raise SystemExit(f"Meta file missing for {latest_feat.name}")

        features_df = pd.read_parquet(latest_feat)
        with meta_path.open() as fh:
            meta = json.load(fh)

        settings = _load_settings()
        k = settings.get("regime_settings", {}).get("cluster_count", 3)

        from systems.regime_cluster import cluster_features

        assignments, centroids, inertia = cluster_features(features_df, meta, k)

        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        assign_path = features_dir / f"regime_assignments_{args.tag}_{timestamp}.csv"
        assignments.to_csv(assign_path, index=False)
        cent_path = features_dir / f"centroids_{args.tag}_{timestamp}.json"
        with cent_path.open("w") as fh:
            json.dump(centroids, fh, indent=2)

        counts = assignments["regime_id"].value_counts().sort_index()
        count_str = " | ".join(
            f"Regime {i}: {c} blocks" for i, c in counts.items()
        )
        print(f"[CLUSTER] K={k} | Inertia={inertia:.2f}")
        print(f"[CLUSTER] {count_str}")
        return

    if not (args.train and args.test and args.step):
        raise SystemExit("Provide --train, --test and --step or use --cluster")

    cache_path = CACHE_DIR / f"{args.tag}_1h.parquet"
    if not cache_path.exists():
        raise SystemExit(
            f"Cache missing for {args.tag}; run fetch-history --fetch-all first"
        )

    df = load_or_fetch(args.tag)
    diffs = df["timestamp"].diff().dropna()
    assert diffs.empty or (diffs == 3600).all(), "Candles must be 1h spaced"

    total = len(df)
    train_c = parse_duration_1h(args.train)
    test_c = parse_duration_1h(args.test)
    step_c = parse_duration_1h(args.step)

    blocks = plan_blocks(df, train_c, test_c, step_c)

    print(
        f"Total={total} | Train={train_c} | Test={test_c} | "
        f"Step={step_c} | Blocks={len(blocks)}"
    )

    if args.verbosity >= 2:
        for idx, b in enumerate(blocks, start=1):
            ts = lambda x: pd.to_datetime(x, unit="s", utc=True)
            print(
                f"Block {idx}: Train {ts(b['train_start'])} -> {ts(b['train_end'])} | "
                f"Test {ts(b['test_start'])} -> {ts(b['test_end'])}"
            )

    logs_dir = Path("logs")
    logs_dir.mkdir(exist_ok=True)
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    plan_path = logs_dir / f"block_plan_{args.tag}_{timestamp}.json"
    with plan_path.open("w") as fh:
        json.dump(blocks, fh, indent=2)
    csv_path = logs_dir / "regime_walk_results.csv"
    with csv_path.open("w") as fh:
        fh.write("block,train_start,train_end,test_start,test_end\n")
    logger.info("Saved block plan to %s", plan_path)

    if args.features:
        from systems.features import FEATURE_NAMES, extract_all_features, save_features

        feat_df = extract_all_features(df, blocks)
        paths = save_features(feat_df, args.tag, timestamp)
        print(
            f"[FEATURES] Extracted {len(FEATURE_NAMES)} features for {len(blocks)} blocks "
            f"-> saved to {Path(paths['raw']).name}"
        )


def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command")
    parser.add_argument("--mode", choices=["fetch-history", "regimes"], help=argparse.SUPPRESS)

    sp_fetch = subparsers.add_parser(
        "fetch-history", help="Fetch and cache 1h candles"
    )
    sp_fetch.add_argument("--tag", required=True, help="Asset tag")
    sp_fetch.add_argument("--fetch-all", action="store_true", help="Fetch full history")
    sp_fetch.add_argument("--start", help="Range start (YYYY-MM-DD)")
    sp_fetch.add_argument("--end", help="Range end (YYYY-MM-DD)")

    sp_regimes = subparsers.add_parser(
        "regimes", help="Walk-forward: Step 1 (plan blocks)"
    )
    sp_regimes.add_argument("--tag", required=True, help="Asset tag")
    sp_regimes.add_argument("--train", help="Training window")
    sp_regimes.add_argument("--test", help="Testing window")
    sp_regimes.add_argument("--step", help="Step size")
    sp_regimes.add_argument(
        "--features",
        action="store_true",
        help="Extract features for training windows",
    )
    sp_regimes.add_argument(
        "--cluster",
        action="store_true",
        help="Run K-Means clustering on extracted features",
    )
    add_verbosity(sp_regimes)

    args = parser.parse_args(argv)

    if getattr(args, "mode", None) and not args.command:
        logger.warning("--mode is deprecated; use subcommands")
        args.command = args.mode

    if not getattr(args, "command", None):
        parser.error("No command provided")

    logging.basicConfig(
        level=max(logging.WARNING - getattr(args, "verbosity", 0) * 10, logging.DEBUG)
    )

    if args.command == "fetch-history":
        cmd_fetch_history(args)
    elif args.command == "regimes":
        cmd_regimes(args)
    else:
        parser.error(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()
