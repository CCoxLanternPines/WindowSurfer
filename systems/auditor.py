from __future__ import annotations

"""Utilities for auditing signal value across features."""

import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

# Features available from teach_engine.enrich_features
FEATURE_COLS = [
    "return",
    "ema_12",
    "ema_26",
    "rsi",
    "volatility",
    "zscore",
    "rolling_high",
    "rolling_low",
]


def _discrete_mi(x: np.ndarray, y: np.ndarray) -> float:
    """Compute mutual information between discrete ``x`` and binary ``y``."""
    if len(x) == 0:
        return 0.0
    crosstab = pd.crosstab(x, y)
    pxy = crosstab / crosstab.values.sum()
    px = pxy.sum(axis=1)
    py = pxy.sum(axis=0)
    mi = 0.0
    for i in pxy.index:
        for j in pxy.columns:
            p = pxy.loc[i, j]
            if p > 0:
                mi += p * np.log2(p / (px.loc[i] * py.loc[j]))
    return float(mi)


def _rank_auc(scores: np.ndarray, labels: np.ndarray) -> float:
    """Simple rank-based ROC-AUC for binary ``labels``."""
    n = len(scores)
    if n == 0:
        return 0.5
    order = np.argsort(scores)
    y_sorted = labels[order]
    pos = y_sorted.sum()
    neg = n - pos
    if pos == 0 or neg == 0:
        return 0.5
    ranks = np.arange(1, n + 1)
    rank_sum = (ranks * y_sorted).sum()
    auc = (rank_sum - pos * (pos + 1) / 2) / (pos * neg)
    return float(auc)


def _js_divergence(p: np.ndarray, q: np.ndarray) -> float:
    """Jensen-Shannon divergence between two histograms."""
    if p.sum() == 0 or q.sum() == 0:
        return 0.0
    p = p / p.sum()
    q = q / q.sum()
    m = 0.5 * (p + q)
    def _kl(a, b):
        mask = (a > 0) & (b > 0)
        return np.sum(a[mask] * np.log2(a[mask] / b[mask]))
    return float(0.5 * (_kl(p, m) + _kl(q, m)))


def audit_brain(
    brain: str,
    df: pd.DataFrame,
    signals: List[Dict],
    use_labels: bool = False,
    horizon: int = 12,
    n_bins: int = 12,
    chart: bool = False,
    symbol: str = "SOLUSD",
    timeframe_tag: str = "",
) -> dict:
    """Run full audit of ``brain`` against ``df`` and ``signals``."""

    N = len(df)
    signal_buy = np.zeros(N, dtype=bool)
    signal_sell = np.zeros(N, dtype=bool)
    signal_any = np.zeros(N, dtype=bool)
    for s in signals:
        idx = s.get("index") or s.get("candle_index")
        if idx is None or idx < 0 or idx >= N:
            continue
        t = s.get("type", "")
        signal_any[idx] = True
        if t == "BUY":
            signal_buy[idx] = True
        elif t == "SELL":
            signal_sell[idx] = True
    if signal_any.sum() == 0:
        print(f"[AUDIT][{brain}] 0 signals in window — falling back to MI/AUC against proxy outcomes using entire window.")

    # Outcomes
    y_pos = np.zeros(N, dtype=int)
    y_neg = np.zeros(N, dtype=int)
    label_path = Path(f"data/labels/{brain}.jsonl")
    label_map: Dict[int, str] = {}
    if label_path.exists() and not use_labels:
        print(f"Found labels for {brain}. Add --use-labels to audit against them.")
    if use_labels and label_path.exists():
        for line in label_path.read_text().splitlines():
            try:
                obj = json.loads(line)
                idx = obj.get("index") or obj.get("candle_index")
                label = obj.get("label")
                if idx is not None and label:
                    label_map[int(idx)] = label
            except Exception:
                continue
        for i in range(N):
            lbl = label_map.get(i)
            if lbl == "BUY":
                y_pos[i] = 1
            elif lbl == "SELL":
                y_neg[i] = 1
        fwd_ret = np.zeros(N)
        use_labels_flag = True
    else:
        fwd_ret = (
            df["close"].shift(-horizon) / df["close"] - 1
        ).to_numpy()
        tau = 0.01
        valid = ~np.isnan(fwd_ret)
        y_pos[valid] = fwd_ret[valid] > tau
        y_neg[valid] = fwd_ret[valid] < -tau
        fwd_ret[np.isnan(fwd_ret)] = 0.0
        use_labels_flag = False

    # Range codex
    range_codex: Dict[str, Dict] = {}
    metrics: Dict[str, Dict] = {}
    weights: Dict[str, float] = {}

    valid_outcome = ~(np.isnan(fwd_ret))

    for feat in FEATURE_COLS:
        series = df[feat].to_numpy()
        mask = ~np.isnan(series) & valid_outcome
        if mask.sum() == 0:
            continue
        x = series[mask]
        y_pos_m = y_pos[mask]
        y_neg_m = y_neg[mask]
        pos_vals = series[(y_pos == 1) & mask]
        neg_vals = series[(y_neg == 1) & mask]
        sig_vals = series[signal_any & mask]
        nonsig_vals = series[(~signal_any) & mask]

        p01 = float(pd.Series(x).quantile(0.01))
        p99 = float(pd.Series(x).quantile(0.99))
        bins = list(pd.Series(x).quantile(np.linspace(0, 1, n_bins + 1)).to_numpy())
        range_codex[feat] = {
            "min": float(np.min(x)),
            "max": float(np.max(x)),
            "robust_min": p01,
            "robust_max": p99,
            "bins": bins,
        }

        bin_idx = np.digitize(x, bins[1:-1], right=True)
        mi_pos = _discrete_mi(bin_idx, y_pos_m)
        mi_neg = _discrete_mi(bin_idx, y_neg_m)
        auc_pos = _rank_auc(x, y_pos_m)

        # Stability across slices
        K = 4
        slice_mis = []
        if len(bin_idx) >= K:
            size = len(bin_idx) // K
            for k in range(K):
                start = k * size
                end = len(bin_idx) if k == K - 1 else (k + 1) * size
                if end > start:
                    slice_mis.append(
                        _discrete_mi(bin_idx[start:end], y_pos_m[start:end])
                    )
        if slice_mis:
            med = float(np.median(slice_mis))
            mn = float(np.min(slice_mis))
            stability = 0.0 if med == 0 else mn / med
        else:
            stability = 0.0

        # Divergences
        hist_pos, _ = np.histogram(pos_vals, bins=bins)
        hist_neg, _ = np.histogram(neg_vals, bins=bins)
        jsd_pos_neg = _js_divergence(hist_pos, hist_neg)
        hist_sig, _ = np.histogram(sig_vals, bins=bins)
        hist_non, _ = np.histogram(nonsig_vals, bins=bins)
        jsd_signal = _js_divergence(hist_sig, hist_non)

        if use_labels_flag:
            baseline = y_pos_m.mean() if len(y_pos_m) else 0.0
            top_mask = x >= np.quantile(x, 0.9)
            top_rate = y_pos_m[top_mask].mean() if top_mask.any() else 0.0
        else:
            baseline = fwd_ret[mask].mean() if mask.any() else 0.0
            top_mask = x >= np.quantile(x, 0.9)
            top_rate = fwd_ret[mask][top_mask].mean() if top_mask.any() else 0.0
        lift_top = float(top_rate - baseline)

        value_score = 0.5 * mi_pos + 0.5 * auc_pos
        net_score = value_score * stability
        metrics[feat] = {
            "lift_top": lift_top,
            "mi_pos": mi_pos,
            "mi_neg": mi_neg,
            "auc_pos": auc_pos,
            "stability": stability,
            "value_score": value_score,
            "jsd_pos_neg": jsd_pos_neg,
            "jsd_signal": jsd_signal,
        }
        weight = float(np.clip(net_score * (1.0 + jsd_pos_neg), 0.0, 1.0))
        weights[feat] = weight

    top_features = sorted(
        ((f, metrics[f]["value_score"] * metrics[f]["stability"]) for f in metrics),
        key=lambda x: x[1],
        reverse=True,
    )
    summary = {"top_features": top_features}

    report = {
        "brain": brain,
        "symbol": symbol,
        "timeframe": timeframe_tag,
        "horizon": horizon,
        "range_codex": range_codex,
        "metrics": metrics,
        "weights": weights,
        "summary": summary,
    }

    audit_path = Path(f"data/audit/{brain}_{symbol}_{timeframe_tag}.json")
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    with audit_path.open("w") as fh:
        json.dump(report, fh, indent=2)

    weights_path = Path(f"data/weights/{brain}.json")
    weights_path.parent.mkdir(parents=True, exist_ok=True)
    with weights_path.open("w") as fh:
        json.dump(weights, fh, indent=2)

    if chart and top_features:
        import matplotlib.pyplot as plt

        names = [n for n, _ in top_features[:10]]
        scores = [s for _, s in top_features[:10]]
        plt.figure(figsize=(10, 5))
        plt.bar(names, scores)
        plt.ylabel("Net Score")
        plt.title(f"Audit {brain} {symbol} {timeframe_tag}")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        plt.show()

    return report
