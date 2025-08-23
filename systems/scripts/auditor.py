from __future__ import annotations

"""Simple weights auditor that ranks features by outcome skew."""

import argparse
import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Iterable

import pandas as pd

from systems.metabrain.engine_utils import (
    BRAIN_MODULES,
    cache_all_brains,
    extract_features_at_t,
)
from systems.metabrain.registry import QUESTION_REGISTRY
from systems.sim_engine import apply_time_filter, parse_timeframe
from systems.utils.regime import compute_regimes, load_regime_settings

ROOT = Path(__file__).resolve().parents[2]
DATA_SIM_PATH = ROOT / "data" / "sim" / "SOLUSD_1h.csv"
WEIGHTS_DIR = ROOT / "data" / "weights"
WEIGHTS_PATH = WEIGHTS_DIR / "weights.json"
BACKUP_PATH = WEIGHTS_DIR / "weights.prev.json"
LOG_PATH = WEIGHTS_DIR / "audit_log.jsonl"


def feature_brain(name: str) -> str:
    if name in QUESTION_REGISTRY:
        return QUESTION_REGISTRY[name][0]
    if "." in name:
        return name.split(".")[0]
    if "_" in name:
        return name.split("_")[0]
    return name


def compute_weights(df: pd.DataFrame, brains: Iterable[str]) -> dict:
    regime_cfg = load_regime_settings()
    regime_df = compute_regimes(df, **regime_cfg)

    brain_cache = cache_all_brains(df)

    records: list[dict] = []
    for t in range(len(df) - 1):
        feats = extract_features_at_t(brain_cache, t)
        trend = regime_df.loc[t, "trend"] if t < len(regime_df) else "unknown"
        vol = regime_df.loc[t, "vol"] if t < len(regime_df) else "unknown"
        feats["regime_key"] = f"{trend}.{vol}"
        feats["label"] = 1 if df["close"].iloc[t + 1] > df["close"].iloc[t] else 0
        records.append(feats)

    data = pd.DataFrame(records).dropna()
    if data.empty:
        return {}

    regimes = data["regime_key"].unique().tolist()
    feature_weights: dict[str, dict[str, float]] = {}
    for feat in [c for c in data.columns if c not in ("label", "regime_key")]:
        if brains and feature_brain(feat) not in brains:
            continue
        feat_data = data[[feat, "label", "regime_key"]].dropna()
        if feat_data.empty:
            continue

        weights_by_regime: dict[str, float] = {}

        # Global weight
        total = len(feat_data)
        weight_global = 0.0
        try:
            bins = pd.qcut(feat_data[feat], 5, duplicates="drop")
        except ValueError:
            bins = None
        if bins is not None:
            for _, grp in feat_data.groupby(bins):
                count = len(grp)
                if count < 20:
                    continue
                rev_pct = grp["label"].mean()
                deviation = rev_pct - 0.5
                sample_fraction = count / total
                weight_global += deviation * sample_fraction
        weights_by_regime["global"] = round(weight_global, 3)

        for reg in regimes:
            sub = feat_data[feat_data["regime_key"] == reg]
            total_reg = len(sub)
            if total_reg < 20:
                continue
            weight_reg = 0.0
            try:
                bins_reg = pd.qcut(sub[feat], 5, duplicates="drop")
            except ValueError:
                continue
            for _, grp in sub.groupby(bins_reg):
                count = len(grp)
                if count < 20:
                    continue
                rev_pct = grp["label"].mean()
                deviation = rev_pct - 0.5
                sample_fraction = count / total_reg
                weight_reg += deviation * sample_fraction
            weights_by_regime[reg] = round(weight_reg, 3)

        feature_weights[feat] = weights_by_regime
        print(f"[AUDITOR] {feat}")
        for reg, w in weights_by_regime.items():
            if reg == "global":
                continue
            sub = feat_data[feat_data["regime_key"] == reg]
            rev_pct = sub["label"].mean() * 100 if len(sub) else 0
            dev = rev_pct - 50
            sign = "+" if dev >= 0 else "-"
            print(f"  {reg}: rev={rev_pct:.0f}% ({sign}{abs(dev):.0f}) weight={w:+.3f}")

    return feature_weights


def run_auditor(brains: Iterable[str], timeframe: str) -> None:
    df = pd.read_csv(DATA_SIM_PATH)
    delta = parse_timeframe(timeframe)
    if delta is not None:
        df = apply_time_filter(df, delta, str(DATA_SIM_PATH))
    df = df.reset_index(drop=True)
    df["candle_index"] = range(len(df))

    weights = compute_weights(df, brains)
    now = datetime.utcnow().replace(microsecond=0).isoformat()
    payload = {
        "version": now,
        "defaults": {"T_buy": 1.0, "T_sell": 1.0},
        "features": weights,
    }

    WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
    if WEIGHTS_PATH.exists():
        shutil.copy(WEIGHTS_PATH, BACKUP_PATH)
    with WEIGHTS_PATH.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)
        fh.write("\n")

    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with LOG_PATH.open("a", encoding="utf-8") as log:
        log.write(json.dumps({"timestamp": now, "brains": list(brains), "time": timeframe}) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Weights Auditor")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--brain", action="append", help="Brain to audit; can repeat")
    group.add_argument("--all", action="store_true", help="Audit all brains")
    parser.add_argument("--time", default="1y", help="Lookback timeframe (e.g. 1y, 6m)")
    args = parser.parse_args()

    brains = BRAIN_MODULES if args.all else args.brain
    run_auditor(brains, args.time)


if __name__ == "__main__":
    main()

