from __future__ import annotations


import argparse
import json
import logging
import sys
import shutil
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

from systems.block_planner import plan_blocks
from systems.cli.common_args import add_verbosity, parse_duration_1h, validate_dates
from systems.data_loader import (
    fetch_all_history_binance,
    fetch_range_kraken,
    load_or_fetch,
)
from systems.paths import (
    ensure_dirs,
    new_run_id,
    raw_parquet,
    temp_blocks_dir,
    temp_features_dir,
    temp_cluster_dir,
    temp_audit_dir,
    results_csv,
    log_file,
    TEMP_DIR,
    temp_run_dir,
)
from systems.utils.cli_args import add_action as cli_add_action
from systems.utils.cli_args import add_run_id as cli_add_run_id
from systems.utils.cli_args import add_tag as cli_add_tag
from systems.utils.cli_args import add_verbose as cli_add_verbose
from contextlib import contextmanager, redirect_stdout, redirect_stderr


class Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for s in self.streams:
            s.write(data)

    def flush(self):
        for s in self.streams:
            s.flush()


@contextmanager
def log_to_file(tag: str, run_id: str):
    ensure_dirs()
    path = log_file(tag, run_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as fh, redirect_stdout(Tee(sys.stdout, fh)), redirect_stderr(
        Tee(sys.stderr, fh)
    ):
        yield

logger = logging.getLogger("bot")


def resolve_ccxt_symbols(tag: str) -> tuple[str, str]:
    path = Path("settings.json")
    settings = {}
    if path.exists():
        with path.open() as fh:
            settings = json.load(fh)
    info = settings.get(tag, {})
    return info.get("kraken_name", tag), info.get("binance_name", tag)


def latest_run_id(tag: str) -> str:
    """Return most recent run id for a given tag."""
    candidates = list(TEMP_DIR.glob(f"*/cluster/centroids_{tag}*.json"))
    if not candidates:
        raise SystemExit(
            f"No clustering artifacts found for tag {tag}; run regimes cluster first"
        )
    latest = max(candidates, key=lambda p: p.stat().st_mtime)
    return latest.parent.parent.name


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
        ensure_dirs()
        cache_path = raw_parquet(args.tag)
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
    cache_path = raw_parquet(args.tag)
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
# Temp purge
# ---------------------------------------------------------------------------


def _parse_age(text: str) -> timedelta:
    if text[-1] not in {"d", "h"}:
        raise ValueError("Use 'd' for days or 'h' for hours")
    value = int(text[:-1])
    if text[-1] == "d":
        return timedelta(days=value)
    return timedelta(hours=value)


def cmd_data_purge_temp(args: argparse.Namespace) -> None:
    ensure_dirs()
    if args.all:
        for d in TEMP_DIR.glob("*"):
            if d.is_dir():
                shutil.rmtree(d)
                print(f"[PURGE] Removed {d}")
        return
    threshold = datetime.utcnow() - _parse_age(args.older_than)
    for d in TEMP_DIR.glob("*"):
        if d.is_dir() and datetime.utcfromtimestamp(d.stat().st_mtime) < threshold:
            shutil.rmtree(d)
            print(f"[PURGE] Removed {d}")

# ---------------------------------------------------------------------------
# Regime utilities
# ---------------------------------------------------------------------------

def regimes_plan(
    tag: str, train: str, test: str, step: str, run_id: str, verbosity: int = 0
) -> None:
    with log_to_file(tag, run_id):
        cache_path = raw_parquet(tag)
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

        blocks_dir = temp_blocks_dir(run_id)
        blocks_dir.mkdir(parents=True, exist_ok=True)
        plan_path = blocks_dir / f"block_plan_{tag}.json"
        with plan_path.open("w") as fh:
            json.dump(blocks, fh, indent=2)
        csv_path = results_csv(tag, run_id)
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        with csv_path.open("w") as fh:
            fh.write("block,train_start,train_end,test_start,test_end\n")
        logger.info("Saved block plan to %s", plan_path)


def regimes_features(tag: str, run_id: str, verbosity: int = 0) -> None:
    with log_to_file(tag, run_id):
        cache_path = raw_parquet(tag)
        if not cache_path.exists():
            raise SystemExit(f"Cache missing for {tag}; run data fetch-history first")
        blocks_path = temp_blocks_dir(run_id) / f"block_plan_{tag}.json"
        if not blocks_path.exists():
            raise SystemExit(f"No block plan found for {tag}; run regimes plan first")
        with blocks_path.open() as fh:
            blocks = json.load(fh)
        df = load_or_fetch(tag)
        from systems.features import FEATURE_NAMES, extract_all_features, save_features

        feat_df = extract_all_features(df, blocks)
        paths = save_features(feat_df, tag, run_id)
        print(
            f"[FEATURES] Extracted {len(FEATURE_NAMES)} features for {len(blocks)} blocks "
            f"-> saved to {Path(paths['raw']).name}"
        )


def regimes_cluster(
    tag: str,
    run_id: str,
    k: int | None = None,
    seed: int | None = None,
    verbosity: int = 0,
) -> None:
    with log_to_file(tag, run_id):
        feat_dir = temp_features_dir(run_id)
        features_path = feat_dir / f"features_{tag}.parquet"
        meta_path = feat_dir / f"features_meta_{tag}.json"
        if not features_path.exists() or not meta_path.exists():
            raise SystemExit(f"No features found for {tag}; run regimes features first")

        features_df = pd.read_parquet(features_path)
        with meta_path.open() as fh:
            meta = json.load(fh)

        from systems.regime_cluster import cluster_features, freeze_brain

        assignments, centroids, inertia = cluster_features(
            features_df, meta, k=k, seed=seed
        )

        cluster_dir = temp_cluster_dir(run_id)
        cluster_dir.mkdir(parents=True, exist_ok=True)
        assign_path = cluster_dir / f"regime_assignments_{tag}.csv"
        assignments.to_csv(assign_path, index=False)
        cent_path = cluster_dir / f"centroids_{tag}.json"
        with cent_path.open("w") as fh:
            json.dump(centroids, fh, indent=2)

        counts = assignments["regime_id"].value_counts().sort_index()
        count_str = " | ".join(f"Regime {i}: {c} blocks" for i, c in counts.items())
        k_used = centroids.get("k", k)
        print(f"[CLUSTER] K={k_used} | Inertia={inertia:.2f}")
        print(f"[CLUSTER] {count_str}")
        freeze_brain(tag, run_id)


# ---------------------------------------------------------------------------
# Audit utilities
# ---------------------------------------------------------------------------

def _resolve_artifacts(tag: str, run_id: Optional[str]) -> Dict[str, Path]:
    search_dirs = []
    if run_id:
        search_dirs.append(temp_run_dir(run_id))
    existing = [d for d in TEMP_DIR.glob("*") if d.is_dir()]
    search_dirs.extend(sorted(existing, key=lambda p: p.stat().st_mtime, reverse=True))

    for base in search_dirs:
        feat = base / "features" / f"features_{tag}.parquet"
        meta = base / "features" / f"features_meta_{tag}.json"
        assign = base / "cluster" / f"regime_assignments_{tag}.csv"
        cent = base / "cluster" / f"centroids_{tag}.json"
        plan = base / "blocks" / f"block_plan_{tag}.json"
        if all(p.exists() for p in [feat, meta, assign, cent, plan]):
            return {
                "features": feat,
                "meta": meta,
                "assignments": assign,
                "centroids": cent,
                "block_plan": plan,
            }

    legacy_feat = Path("features")
    if legacy_feat.exists():
        print("[DEPRECATED] Legacy layout detected; migrating...")
        try:
            from scripts.migrate_layout import main as _migrate

            _migrate()
        except Exception as exc:
            print(f"[MIGRATE][WARN] {exc}")
        return _resolve_artifacts(tag, run_id)

    raise SystemExit(
        "Missing artifacts for audit; run regimes features and cluster first"
    )


def cmd_audit_summary(args: argparse.Namespace) -> None:
    run_id = args.run_id
    paths = _resolve_artifacts(args.tag, run_id)
    from systems.regime_audit import run_audit

    with log_to_file(args.tag, run_id):
        run_audit(args.tag, paths, run_id, args.verbosity)


def cmd_audit_full(args: argparse.Namespace) -> None:
    run_id = args.run_id
    feat_dir = temp_features_dir(run_id)
    features_path = feat_dir / f"features_{args.tag}.parquet"
    meta_path = feat_dir / f"features_meta_{args.tag}.json"
    if not (features_path.exists() and meta_path.exists()):
        regimes_features(args.tag, run_id, args.verbosity)
    cluster_dir = temp_cluster_dir(run_id)
    assign_path = cluster_dir / f"regime_assignments_{args.tag}.csv"
    cent_path = cluster_dir / f"centroids_{args.tag}.json"
    if not (assign_path.exists() and cent_path.exists()):
        regimes_cluster(args.tag, run_id, verbosity=args.verbosity)
    paths = _resolve_artifacts(args.tag, run_id)
    from systems.regime_audit import run_audit
    from systems.regime_audit_plus import run as run_plus

    with log_to_file(args.tag, run_id):
        run_audit(args.tag, paths, run_id, args.verbosity)
        run_plus(args.tag, paths, run_id, args.verbosity)


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

    sp_purge = data_sub.add_parser("purge-temp", help="Purge temporary run data")
    sp_purge.add_argument("--older-than", default="7d", help="Age threshold, e.g. 7d")
    sp_purge.add_argument("--all", action="store_true", help="Remove all temp runs")

    # Regimes group
    sp_regimes = subparsers.add_parser(
        "regimes", help="Regime training/assign/audit/tune"
    )
    cli_add_action(sp_regimes, choices=["train", "assign", "audit", "tune"])
    cli_add_tag(sp_regimes, required=False)
    cli_add_run_id(sp_regimes, required=False)
    cli_add_verbose(sp_regimes)
    sp_regimes.add_argument("--regime-id", type=int, help="Regime identifier")
    sp_regimes.add_argument("--tau", type=float, default=0.70, help="Purity threshold")
    sp_regimes.add_argument("--trials", type=int, default=50, help="Optuna trials")
    sp_regimes.add_argument("--metric", type=str, default="pnl_dd", help="Optimization metric")
    sp_regimes.add_argument("--seed", type=int, default=2, help="RNG seed")
    sp_regimes.add_argument("--smoke", action="store_true", help=argparse.SUPPRESS)

    reg_sub = sp_regimes.add_subparsers(dest="command")

    sp_plan = reg_sub.add_parser("plan", help="Plan walk-forward blocks")
    sp_plan.add_argument("--tag", required=True, help="Asset tag")
    sp_plan.add_argument("--train", required=True, help="Training window")
    sp_plan.add_argument("--test", required=True, help="Testing window")
    sp_plan.add_argument("--step", required=True, help="Step size")
    cli_add_run_id(sp_plan)
    add_verbosity(sp_plan)

    sp_feat = reg_sub.add_parser("features", help="Extract features for blocks")
    sp_feat.add_argument("--tag", required=True, help="Asset tag")
    cli_add_run_id(sp_feat)
    add_verbosity(sp_feat)

    sp_clust = reg_sub.add_parser("cluster", help="Run K-Means clustering on features")
    sp_clust.add_argument("--tag", required=True, help="Asset tag")
    cli_add_run_id(sp_clust)
    sp_clust.add_argument("--k", type=int, help="Number of clusters")
    sp_clust.add_argument("--seed", type=int, help="RNG seed")
    add_verbosity(sp_clust)

    sp_purity = reg_sub.add_parser("purity", help="Estimate regime purity for blocks")
    sp_purity.add_argument("--tag", required=True, help="Asset tag")
    cli_add_run_id(sp_purity, required=True)
    sp_purity.add_argument("--tau", type=float, default=0.70, help="Purity threshold")
    sp_purity.add_argument("--win", default="1w", help="Sub-window duration")
    sp_purity.add_argument("--stride", type=int, default=6, help="Stride in candles")

    # Audit group
    sp_audit = subparsers.add_parser("audit", help="Audit regimes")
    audit_sub = sp_audit.add_subparsers(dest="command", required=True)

    sp_sum = audit_sub.add_parser("summary", help="Run audit summary")
    sp_sum.add_argument("--tag", required=True, help="Asset tag")
    cli_add_run_id(sp_sum)
    add_verbosity(sp_sum)

    sp_full = audit_sub.add_parser("full", help="Run full audit pipeline")
    sp_full.add_argument("--tag", required=True, help="Asset tag")
    cli_add_run_id(sp_full)
    add_verbosity(sp_full)

    sp_brain_final = audit_sub.add_parser("brain", help="Finalize brain artifact")
    sp_brain_final.add_argument("--tag", required=True, help="Asset tag")
    cli_add_run_id(sp_brain_final)
    sp_brain_final.add_argument(
        "--labels", help="Path to JSON mapping regime id to label"
    )
    sp_brain_final.add_argument("--alpha", type=float, default=0.2)
    sp_brain_final.add_argument("--switch-margin", type=float, default=0.3)

    # Brain group
    sp_brain = subparsers.add_parser("brain", help="Brain utilities")
    brain_sub = sp_brain.add_subparsers(dest="command", required=True)

    sp_classify = brain_sub.add_parser("classify", help="Classify current regime")
    sp_classify.add_argument("--tag", required=True, help="Asset tag")
    sp_classify.add_argument("--train", required=True, help="Training window")
    sp_classify.add_argument("--at", help="ISO timestamp")
    sp_classify.add_argument("--from-csv", dest="from_csv", help="Path to CSV of candles")

    args = parser.parse_args(argv)

    verbosity = getattr(args, "verbosity", getattr(args, "verbose", 0))
    logging.basicConfig(
        level=max(logging.WARNING - verbosity * 10, logging.DEBUG)
    )

    if args.group == "data":
        if args.command == "fetch-history":
            cmd_data_fetch(args)
        elif args.command == "verify":
            cmd_data_verify(args)
        elif args.command == "purge-temp":
            cmd_data_purge_temp(args)
    elif args.group == "regimes":
        if args.action == "tune":
            from systems.regime_tuner import run_regime_tuning

            run_regime_tuning(
                tag=args.tag,
                run_id=(args.run_id or "default"),
                regime_id=args.regime_id,
                tau=args.tau,
                trials=args.trials,
                metric=args.metric,
                seed=args.seed,
                verbose=args.verbose,
                smoke=args.smoke,
            )
            return
        run_id = args.run_id or new_run_id("regimes")
        if args.command == "plan":
            regimes_plan(args.tag, args.train, args.test, args.step, run_id, args.verbosity)
        elif args.command == "features":
            regimes_features(args.tag, run_id, args.verbosity)
        elif args.command == "cluster":
            regimes_cluster(
                args.tag, run_id, k=args.k, seed=args.seed, verbosity=args.verbosity
            )
        elif args.command == "purity":
            from systems.purity import compute_purity

            compute_purity(
                tag=args.tag,
                run_id=run_id,
                tau=args.tau,
                win_dur=args.win,
                stride=args.stride,
            )
    elif args.group == "audit":
        if args.command == "brain":
            from systems.brain import finalize_brain, write_latest_copy

            run_id = args.run_id or latest_run_id(args.tag)

            labels = None
            if args.labels:
                with open(args.labels) as fh:
                    raw = json.load(fh)
                labels = {int(k): v for k, v in raw.items()}

            path = finalize_brain(
                args.tag, run_id, labels, args.alpha, args.switch_margin
            )
            latest = write_latest_copy(path, args.tag)
            brain = json.loads(Path(path).read_text())
            row_sums = [round(sum(r), 6) for r in brain["transitions"]]
            print(f"[BRAIN] Saved {path} and updated {latest}")
            print(f"[BRAIN] Row sums: {row_sums}")
            print(f"[BRAIN] Labels: {brain.get('labels', {})}")
        else:
            run_id = args.run_id or new_run_id("regimes")
            args.run_id = run_id
            if args.command == "summary":
                cmd_audit_summary(args)
            elif args.command == "full":
                cmd_audit_full(args)
    elif args.group == "brain":
        if args.command == "classify":
            from systems.live_classifier import classify

            train_candles = parse_duration_1h(args.train)
            res = classify(
                tag=args.tag,
                train_candles=train_candles,
                at_ts=args.at,
                csv_path=args.from_csv,
            )
            print(
                f"[CLASSIFY] regime_id={res['regime_id']} | probs_next={res['probs_next']} | features={res['features_used']}"
            )
    else:
        parser.error("Unknown command")


if __name__ == "__main__":
    main()

