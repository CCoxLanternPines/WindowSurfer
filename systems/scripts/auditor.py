from __future__ import annotations

"""Export per-candle feature statistics for brains."""

import argparse
import json
import sys
from pathlib import Path
from typing import Iterable

SCRIPTS_DIR = Path(__file__).resolve().parent
sys.path = [p for p in sys.path if Path(p).resolve() != SCRIPTS_DIR]
sys.modules.pop("math", None)
import importlib
sys.modules["math"] = importlib.import_module("math")

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import pandas as pd
import numpy as np

from systems.metabrain.engine_utils import (
    BRAIN_MODULES,
    cache_all_brains,
    extract_features_at_t,
)
from systems.metabrain.registry import QUESTION_REGISTRY
from systems.sim_engine import apply_time_filter, parse_timeframe
from systems.utils.regime import compute_regimes, load_regime_settings

DATA_SIM_PATH = ROOT / "data" / "sim" / "SOLUSD_1h.csv"
DEFAULT_OUT_DIR = ROOT / "data" / "stats"


def to_native(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.ndarray,)):
        return o.tolist()
    return o


def feature_brain(name: str) -> str:
    if name in QUESTION_REGISTRY:
        return QUESTION_REGISTRY[name][0]
    if "." in name:
        return name.split(".")[0]
    if "_" in name:
        return name.split("_")[0]
    return name


def export_stats(brains: Iterable[str], timeframe: str, out_dir: Path) -> None:
    df = pd.read_csv(DATA_SIM_PATH)
    delta = parse_timeframe(timeframe)
    if delta is not None:
        df = apply_time_filter(df, delta, str(DATA_SIM_PATH))
    df = df.reset_index(drop=True)
    df["candle_index"] = range(len(df))

    regime_cfg = load_regime_settings()
    regime_df = compute_regimes(df, **regime_cfg)

    brain_cache = cache_all_brains(df)

    out_dir.mkdir(parents=True, exist_ok=True)

    for brain in brains:
        out_jsonl = out_dir / f"{brain}_{timeframe}.jsonl"
        summary_path = out_dir / f"{brain}_{timeframe}_summary.json"

        rows: list[dict] = []
        for t in range(len(df)):
            features_all = extract_features_at_t(brain_cache, t)
            features_brain = {
                k: v for k, v in features_all.items() if feature_brain(k) == brain
            }
            regime = {
                "trend": regime_df.loc[t, "trend"] if t < len(regime_df) else "unknown",
                "vol": regime_df.loc[t, "vol"] if t < len(regime_df) else "unknown",
            }
            row = {
                "idx": int(t),
                "timestamp": df["timestamp"].iloc[t],
                "price": float(df["close"].iloc[t]),
                "regime": regime,
                "features": features_brain,
            }
            rows.append(row)

        with out_jsonl.open("w", encoding="utf-8") as fh:
            for r in rows:
                fh.write(json.dumps(r, default=to_native) + "\n")

        summary: dict[str, float] = {}
        if rows:
            feats_df = pd.DataFrame([r["features"] for r in rows])
            for col in feats_df.columns:
                series = pd.to_numeric(feats_df[col], errors="coerce").dropna()
                if series.empty:
                    continue
                if "slowdown_ratio" in col:
                    summary["avg_slowdown_ratio"] = float(series.mean())
                    summary["collapse_rate"] = float((series < 0.3).mean())
                elif "lead_time" in col:
                    summary["lead_time_mean"] = float(series.mean())
                elif "resolution_rev" in col:
                    summary["resolution_rev"] = float(series.mean())
                elif "resolution_cont" in col:
                    summary["resolution_cont"] = float(series.mean())
                else:
                    summary[f"avg_{col}"] = float(series.mean())

        payload = {"brain": brain, "time": timeframe, "summary": summary}
        with summary_path.open("w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2)
            fh.write("\n")

        print(
            f"[AUDITOR] Exporting stats for {brain} ({timeframe}) → {out_jsonl.relative_to(ROOT)}"
        )
        feature_count = len(feats_df.columns) if rows else 0
        print(f"  {len(rows)} candles processed, {feature_count} features exported")
        if "avg_slowdown_ratio" in summary:
            pct = round(summary.get("collapse_rate", 0) * 100)
            lead = summary.get("lead_time_mean", 0)
            print(
                f"  Summary: slowdown_ratio collapse<0.3 = {pct}%, lead_time={lead:.1f}c"
            )

    print("[AUDITOR] Done")


def main() -> None:
    parser = argparse.ArgumentParser(description="Statistics Exporter")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--brain", action="append", help="Brain to export; can repeat")
    group.add_argument("--all", action="store_true", help="Export all brains")
    parser.add_argument("--time", default="1y", help="Lookback timeframe (e.g. 1y, 6m)")
    parser.add_argument("--out", default=str(DEFAULT_OUT_DIR), help="Output directory")
    args = parser.parse_args()

    brains = BRAIN_MODULES if args.all else args.brain
    export_stats(brains, args.time, Path(args.out))


if __name__ == "__main__":
    main()

