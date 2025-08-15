from __future__ import annotations

"""Position-based buy evaluation."""

from typing import Dict, Any

from systems.utils.addlog import addlog

# ==== CONSENSUS TUNING (adjust here; no settings changes) =====================

# Windows used for consensus. We compute min/max/mid/pos/slope at each.
# Span strings are parsed against the candle interval detected from the series.
CONSENSUS_WINDOWS = [
    ("D",  "1d",  1.00),   # weight
    ("W",  "7d",  0.70),
    ("M", "30d",  0.50),
    ("Q", "90d",  0.40),
]

# Percentile thresholds (computed from rolling history of scores)
BOTTOM_Q = 0.85     # buy threshold
TOP_Q    = 0.85     # sell threshold
LOOKBACK_FOR_Q = "90d"    # window to compute percentiles over (rolling)

# Turning filter on smallest frame (first entry in CONSENSUS_WINDOWS)
TURN_EMA_FRAC = 0.10  # slope EMA span = frac * bars_in_window

# Target construction for sells (attached to note at buy time)
ALPHA_H_TOP = 0.60    # weight for day headroom
BETA_H_TOP  = 0.40    # weight for week/month headroom (averaged)

# Safety
COOLDOWN_MOVE_FRAC = 0.25  # price must move this fraction of day width from last buy

# ==============================================================================


def _parse_span_seconds(s: str) -> int:
    # supports “Xm”, “Xh”, “Xd”, “Xw”; lower/upper case
    n = int("".join(ch for ch in s if ch.isdigit()))
    u = "".join(ch for ch in s if ch.isalpha()).lower()
    mult = {"m":60, "h":3600, "d":86400, "w":604800}[u]
    return n * mult


def _detect_step_seconds(series) -> int:
    # assumes monotonically increasing; uses median diff for robustness
    import numpy as np

    ts = series["timestamp"].to_numpy()
    if ts.size < 3:
        return 3600
    diffs = np.diff(ts[-200:]) if ts.size > 200 else np.diff(ts)
    return int(np.median(diffs)) or 3600


def _bars_for_span(series, span_str: str) -> int:
    step = _detect_step_seconds(series)
    return max(2, _parse_span_seconds(span_str) // step)


def _window_features(series, t: int, bars: int) -> dict:
    """Return low/high/mid/width/pos/slope/vol/hTop/hBot at index t for a window size in bars."""
    import numpy as np
    import pandas as pd

    lo = max(0, t - bars + 1)
    window = series.iloc[lo:t+1]
    if window.shape[0] < 2:
        px = float(series.iloc[t]["close"])
        return dict(low=px, high=px, mid=px, width=1e-9, pos=0.0, slope=0.0, vol=0.0, hTop=0.0, hBot=0.0)
    price = window["close"].to_numpy()
    low, high = float(price.min()), float(price.max())
    width = max(high - low, 1e-9)
    mid = (low + high) / 2.0
    pos = (float(price[-1]) - mid) / (width/2.0)
    pos = min(1.0, max(-1.0, pos))
    # slope: EMA of deltas over ~10% of bars
    span = max(2, int(bars * TURN_EMA_FRAC))
    deltas = pd.Series(price).diff().fillna(0.0)
    slope = float(deltas.ewm(span=span, adjust=False).mean().iloc[-1]) / width
    # volatility proxy: median abs deviation over window, scaled by price
    med = float(pd.Series(price).median())
    mad = float((pd.Series(price) - med).abs().median())
    vol = mad / max(med, 1e-9)
    now = float(price[-1])
    hTop = max(high - now, 0.0) / max(now, 1e-9)
    hBot = max(now - low, 0.0) / max(now, 1e-9)
    return dict(low=low, high=high, mid=mid, width=width, pos=pos, slope=slope, vol=vol, hTop=hTop, hBot=hBot)


def _consensus(series, t: int):
    """Compute BottomScore, TopScore, feature dicts per window, and turning flags."""
    # determine bars for each configured span (drop windows that don’t fit)
    spans = []
    for _, span_str, w in CONSENSUS_WINDOWS:
        bars = _bars_for_span(series, span_str)
        if t+1 >= bars:
            spans.append((span_str, bars, w))
    # compute features
    feats = []
    for span_str, bars, w in spans:
        f = _window_features(series, t, bars)
        f["span_str"], f["bars"], f["weight"] = span_str, bars, w
        feats.append(f)
    if not feats:
        return 0.0, 0.0, [], False, False

    # first window is the trigger frame
    f0 = feats[0]
    bottom = sum(f["weight"] * max(-f["pos"], 0.0) for f in feats)
    top    = sum(f["weight"] * max( f["pos"], 0.0) for f in feats)
    turning_up   = f0["slope"] > 0.0
    turning_down = f0["slope"] <= 0.0
    return bottom, top, feats, turning_up, turning_down


def _score_cache(state: dict) -> dict:
    """Stateful rolling cache to compute percentiles without recomputing full history."""
    key = "consensus_cache"
    if key not in state:
        state[key] = {"bottom": [], "top": [], "ts": []}
    return state[key]


def _bars_for_lookback(series, lookback_str: str) -> int:
    return _bars_for_span(series, lookback_str)


def _percentile(values, q: float) -> float:
    import numpy as np

    if not values:
        return float("inf") if q >= 0.5 else float("-inf")
    return float(np.quantile(np.array(values), q))


def evaluate_buy(
    ctx: Dict[str, Any],
    t: int,
    series,
    *,
    window_name: str,
    cfg: Dict[str, Any],
    runtime_state: Dict[str, Any],
):
    """Return sizing and metadata for a buy signal in ``window_name``."""

    verbose = runtime_state.get("verbose", 0)
    candle = series.iloc[t]

    bottom, top, feats, turning_up, _turning_down = _consensus(series, t)

    runtime_state = ctx.get("runtime_state", runtime_state)
    cache = _score_cache(runtime_state)
    cache["bottom"].append(bottom)
    cache["top"].append(top)
    if "timestamp" in series.columns:
        cache["ts"].append(int(series.iloc[t]["timestamp"]))

    lb = _bars_for_lookback(series, LOOKBACK_FOR_Q)
    bottom_hist = cache["bottom"][-lb:] if lb > 0 else cache["bottom"]
    thresh_buy = _percentile(bottom_hist, BOTTOM_Q)

    last_buy_idx = runtime_state.get(f"last_buy_idx::{window_name}")
    ok_move = True
    if last_buy_idx is not None:
        f0 = feats[0] if feats else {"width": 0.0}
        moved = abs(float(series.iloc[t]["close"]) - float(series.iloc[last_buy_idx]["close"]))
        ok_move = moved >= COOLDOWN_MOVE_FRAC * max(f0.get("width", 0.0), 1e-9)

    addlog(
        f"[BUY?][{window_name} {cfg['window_size']}] bottom={bottom:.3f} ≥ {thresh_buy:.3f} turn↑={turning_up}",
        verbose_int=3,
        verbose_state=ctx.get("verbose"),
    )

    should_buy = (bottom >= thresh_buy) and turning_up and ok_move
    if not should_buy:
        return False

    capital = runtime_state.get("capital", 0.0)
    base = cfg.get("investment_fraction", 0.0)
    size_usd = capital * base

    limits = runtime_state.get("limits", {})
    min_sz = float(limits.get("min_note_size", 0.0))
    max_sz = float(limits.get("max_note_usdt", float("inf")))
    raw = size_usd
    size_usd = min(size_usd, capital, max_sz)
    if raw != size_usd:
        addlog(
            f"[CLAMP] size=${raw:.2f} → ${size_usd:.2f} (cap=${capital:.2f}, max=${max_sz:.2f})",
            verbose_int=2,
            verbose_state=verbose,
        )
    if size_usd < min_sz:
        addlog(
            f"[SKIP][{window_name} {cfg['window_size']}] size=${size_usd:.2f} < min=${min_sz:.2f}",
            verbose_int=2,
            verbose_state=verbose,
        )
        return False

    sz_pct = (size_usd / capital * 100) if capital else 0.0

    price_now = float(series.iloc[t]["close"])
    if feats:
        f0 = feats[0]
        bigger = [f for f in feats[1:]]
        hTop_big = (sum(f["hTop"] for f in bigger) / len(bigger)) if bigger else f0["hTop"]
        expected_up = ALPHA_H_TOP * f0["hTop"] + BETA_H_TOP * hTop_big
        vol_clip = min(1.3, max(0.7, f0["vol"] / (1e-9 + f0["vol"])))
        target_roi = expected_up * vol_clip
        target_price = price_now * (1.0 + max(0.0, target_roi))
    else:
        target_price = price_now * 1.02

    meta = {
        "consensus_bottom": bottom,
        "consensus_top": top,
        "turning_up": turning_up,
        "target_price": float(target_price),
    }

    addlog(
        f"[BUY][{window_name} {cfg['window_size']}] bottom={bottom:.3f}, base={base*100:.2f}% → size={sz_pct:.2f}% (cap=${size_usd:.2f})",
        verbose_int=1,
        verbose_state=verbose,
    )

    result = {
        "size_usd": size_usd,
        "window_name": window_name,
        "window_size": cfg["window_size"],
        **meta,
    }
    if "timestamp" in series.columns:
        result["created_ts"] = int(candle["timestamp"])
    result["created_idx"] = t

    runtime_state[f"last_buy_idx::{window_name}"] = t

    return result
