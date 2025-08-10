from __future__ import annotations


import argparse
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

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
    path = Path("settings.json")
    settings = {}
    if path.exists():
        with path.open() as fh:
            settings = json.load(fh)
    info = settings.get(tag, {})
    return info.get("kraken_name", tag), info.get("binance_name", tag)


# ---------------------------------------------------------------------------
# Data commands
# ---------------------------------------------------------------------------

def cmd_data_fetch(args: argparse.Namespace) -> None:
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


def cmd_data_verify(args: argparse.Namespace) -> None:
    cache_path = CACHE_DIR / f"{args.tag}_1h.parquet"
    if not cache_path.exists():
        raise SystemExit(f"Cache missing for {args.tag}; run data fetch-history first")
    df = pd.read_parquet(cache_path)
    ts = pd.to_datetime(df["timestamp"], unit="s", utc=True)
    diffs = ts.diff().dropna().dt.total_seconds()
    gaps = int((diffs != 3600).sum())
    first = ts.iloc[0] if not df.empty else None
    last = ts.iloc[-1] if not df.empty else None
    print(f"[VERIFY] TAG={args.tag} | Range={first} -> {last} | gaps={gaps}")


# ---------------------------------------------------------------------------
# Regime utilities
# ---------------------------------------------------------------------------

def regimes_plan(tag: str, train: str, test: str, step: str, verbosity: int = 0) -> None:
    cache_path = CACHE_DIR / f"{tag}_1h.parquet"
    if not cache_path.exists():
        raise SystemExit(f"Cache missing for {tag}; run data fetch-history first")
    df = load_or_fetch(tag)
    diffs = df["timestamp"].diff().dropna()
    assert diffs.empty or (diffs == 3600).all(), "Candles must be 1h spaced"

    total = len(df)
    train_c = parse_duration_1h(train)
    test_c = parse_duration_1h(test)
    step_c = parse_duration_1h(step)

    blocks = plan_blocks(df, train_c, test_c, step_c)

    print(
        f"Total={total} | Train={train_c} | Test={test_c} | "
        f"Step={step_c} | Blocks={len(blocks)}"
    )

    if verbosity >= 2:
        for idx, b in enumerate(blocks, start=1):
            ts = lambda x: pd.to_datetime(x, unit="s", utc=True)
            print(
                f"Block {idx}: Train {ts(b['train_start'])} -> {ts(b['train_end'])} | "
                f"Test {ts(b['test_start'])} -> {ts(b['test_end'])}"
            )

    logs_dir = Path("logs")
    logs_dir.mkdir(exist_ok=True)
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    plan_path = logs_dir / f"block_plan_{tag}_{timestamp}.json"
    with plan_path.open("w") as fh:
        json.dump(blocks, fh, indent=2)
    csv_path = logs_dir / "regime_walk_results.csv"
    with csv_path.open("w") as fh:
        fh.write("block,train_start,train_end,test_start,test_end\n")
    logger.info("Saved block plan to %s", plan_path)


def regimes_features(tag: str, verbosity: int = 0) -> None:
    cache_path = CACHE_DIR / f"{tag}_1h.parquet"
    if not cache_path.exists():
        raise SystemExit(f"Cache missing for {tag}; run data fetch-history first")
    plan_files = sorted(Path("logs").glob(f"block_plan_{tag}_*.json"))
    if not plan_files:
        raise SystemExit(f"No block plan found for {tag}; run regimes plan first")
    with plan_files[-1].open() as fh:
        blocks = json.load(fh)
    df = load_or_fetch(tag)
    from systems.features import FEATURE_NAMES, extract_all_features, save_features

    feat_df = extract_all_features(df, blocks)
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    paths = save_features(feat_df, tag, timestamp)
    print(
        f"[FEATURES] Extracted {len(FEATURE_NAMES)} features for {len(blocks)} blocks "
        f"-> saved to {Path(paths['raw']).name}"
    )


def regimes_cluster(tag: str, verbosity: int = 0) -> None:
    features_dir = Path("features")
    feat_files = sorted(features_dir.glob(f"features_{tag}_*.parquet"))
    if not feat_files:
        raise SystemExit(f"No features found for {tag}; run regimes features first")
    latest_feat = feat_files[-1]
    stamp = "_".join(latest_feat.stem.split("_")[2:])
    meta_path = features_dir / f"features_meta_{tag}_{stamp}.json"
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
    assign_path = features_dir / f"regime_assignments_{tag}_{timestamp}.csv"
    assignments.to_csv(assign_path, index=False)
    cent_path = features_dir / f"centroids_{tag}_{timestamp}.json"
    with cent_path.open("w") as fh:
        json.dump(centroids, fh, indent=2)

    counts = assignments["regime_id"].value_counts().sort_index()
    count_str = " | ".join(f"Regime {i}: {c} blocks" for i, c in counts.items())
    print(f"[CLUSTER] K={k} | Inertia={inertia:.2f}")
    print(f"[CLUSTER] {count_str}")


# ---------------------------------------------------------------------------
# Audit utilities
# ---------------------------------------------------------------------------

def _resolve_latest_artifacts(tag: str) -> Dict[str, Path]:
    features_dir = Path("features")
    logs_dir = Path("logs")
    feat_files = sorted(features_dir.glob(f"features_{tag}_*.parquet"))
    meta_files = sorted(features_dir.glob(f"features_meta_{tag}_*.json"))
    assign_files = sorted(features_dir.glob(f"regime_assignments_{tag}_*.csv"))
    cent_files = sorted(features_dir.glob(f"centroids_{tag}_*.json"))
    plan_files = sorted(logs_dir.glob(f"block_plan_{tag}_*.json"))
    if not (feat_files and meta_files and assign_files and cent_files and plan_files):
        raise SystemExit(
            "Missing artifacts for audit; run regimes features and cluster first"
        )
    return {
        "features": feat_files[-1],
        "meta": meta_files[-1],
        "assignments": assign_files[-1],
        "centroids": cent_files[-1],
        "block_plan": plan_files[-1],
    }


def cmd_audit_summary(args: argparse.Namespace) -> None:
    paths = _resolve_latest_artifacts(args.tag)
    from systems.regime_audit import run_audit

    run_audit(args.tag, paths, args.verbosity)


def cmd_audit_full(args: argparse.Namespace) -> None:
    features_dir = Path("features")
    feat_files = sorted(features_dir.glob(f"features_{args.tag}_*.parquet"))
    meta_files = sorted(features_dir.glob(f"features_meta_{args.tag}_*.json"))
    if not (feat_files and meta_files):
        regimes_features(args.tag, args.verbosity)
    assign_files = sorted(features_dir.glob(f"regime_assignments_{args.tag}_*.csv"))
    cent_files = sorted(features_dir.glob(f"centroids_{args.tag}_*.json"))
    if not (assign_files and cent_files):
        regimes_cluster(args.tag, args.verbosity)
    paths = _resolve_latest_artifacts(args.tag)
    from systems.regime_audit import run_audit
    from systems.regime_audit_plus import run as run_plus

    run_audit(args.tag, paths, args.verbosity)
    run_plus(args.tag, args.verbosity)


# ---------------------------------------------------------------------------
# Main CLI
# ---------------------------------------------------------------------------

def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="group", required=True)

    # Data group
    sp_data = subparsers.add_parser("data", help="Data management")
    data_sub = sp_data.add_subparsers(dest="command", required=True)

    sp_fetch = data_sub.add_parser("fetch-history", help="Fetch and cache 1h candles")
    sp_fetch.add_argument("--tag", required=True, help="Asset tag")
    sp_fetch.add_argument("--fetch-all", action="store_true", help="Fetch full history")
    sp_fetch.add_argument("--start", help="Range start (YYYY-MM-DD)")
    sp_fetch.add_argument("--end", help="Range end (YYYY-MM-DD)")
    add_verbosity(sp_fetch)

    sp_verify = data_sub.add_parser("verify", help="Verify cached data continuity")
    sp_verify.add_argument("--tag", required=True, help="Asset tag")
    add_verbosity(sp_verify)

    # Regimes group
    sp_regimes = subparsers.add_parser("regimes", help="Regime workflows")
    reg_sub = sp_regimes.add_subparsers(dest="command", required=True)

    sp_plan = reg_sub.add_parser("plan", help="Plan walk-forward blocks")
    sp_plan.add_argument("--tag", required=True, help="Asset tag")
    sp_plan.add_argument("--train", required=True, help="Training window")
    sp_plan.add_argument("--test", required=True, help="Testing window")
    sp_plan.add_argument("--step", required=True, help="Step size")
    add_verbosity(sp_plan)

    sp_feat = reg_sub.add_parser("features", help="Extract features for blocks")
    sp_feat.add_argument("--tag", required=True, help="Asset tag")
    add_verbosity(sp_feat)

    sp_clust = reg_sub.add_parser("cluster", help="Run K-Means clustering on features")
    sp_clust.add_argument("--tag", required=True, help="Asset tag")
    add_verbosity(sp_clust)

    # Audit group
    sp_audit = subparsers.add_parser("audit", help="Audit regimes")
    audit_sub = sp_audit.add_subparsers(dest="command", required=True)

    sp_sum = audit_sub.add_parser("summary", help="Run audit summary")
    sp_sum.add_argument("--tag", required=True, help="Asset tag")
    add_verbosity(sp_sum)

    sp_full = audit_sub.add_parser("full", help="Run full audit pipeline")
    sp_full.add_argument("--tag", required=True, help="Asset tag")
    add_verbosity(sp_full)

    args = parser.parse_args(argv)

    logging.basicConfig(
        level=max(logging.WARNING - getattr(args, "verbosity", 0) * 10, logging.DEBUG)
    )

    if args.group == "data":
        if args.command == "fetch-history":
            cmd_data_fetch(args)
        elif args.command == "verify":
            cmd_data_verify(args)
    elif args.group == "regimes":
        if args.command == "plan":
            regimes_plan(args.tag, args.train, args.test, args.step, args.verbosity)
        elif args.command == "features":
            regimes_features(args.tag, args.verbosity)
        elif args.command == "cluster":
            regimes_cluster(args.tag, args.verbosity)
    elif args.group == "audit":
        if args.command == "summary":
            cmd_audit_summary(args)
        elif args.command == "full":
            cmd_audit_full(args)
    else:
        parser.error("Unknown command")


if __name__ == "__main__":
    main()

