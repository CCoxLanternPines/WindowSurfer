from __future__ import annotations

"""Price-target-based sell evaluation."""

from typing import Any, Dict, List

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


def evaluate_sell(
    ctx: Dict[str, Any],
    t: int,
    series,
    *,
    window_name: str,
    cfg: Dict[str, Any],
    open_notes: List[Dict[str, Any]],
    runtime_state: Dict[str, Any] | None = None,
) -> List[Dict[str, Any]]:
    """Return a list of notes to sell in ``window_name`` on this candle."""

    verbose = 0
    if runtime_state:
        verbose = runtime_state.get("verbose", 0)

    candle = series.iloc[t]
    price = float(candle["close"])

    bottom, top, feats, _turning_up, turning_down = _consensus(series, t)
    runtime_state = ctx.get("runtime_state", runtime_state or {})
    cache = _score_cache(runtime_state)
    cache["bottom"].append(bottom)
    cache["top"].append(top)
    lb = _bars_for_lookback(series, LOOKBACK_FOR_Q)
    top_hist = cache["top"][-lb:] if lb > 0 else cache["top"]
    thresh_sell = _percentile(top_hist, TOP_Q)

    window_notes = [n for n in open_notes if n.get("window_name") == window_name]
    open_count = len(window_notes)

    ledger = ctx.get("ledger") if ctx else None
    closed_count = 0
    if ledger:
        closed_count = sum(
            1 for n in ledger.get_closed_notes() if n.get("window_name") == window_name
        )

    future_targets = [
        n.get("target_price", float("inf"))
        for n in window_notes
        if n.get("target_price", float("inf")) > price
    ]
    next_target = min(future_targets) if future_targets else None

    candidates = window_notes

    def roi_now(note: Dict[str, Any]) -> float:
        buy = note.get("entry_price", 0.0)
        return (price - buy) / buy if buy else 0.0

    candidates.sort(key=roi_now, reverse=True)

    state = {
        "sell_count": 0,
        "verbose": verbose,
        "window_name": window_name,
        "window_size": cfg["window_size"],
        "max_sells": cfg.get("max_notes_sell_per_candle", 1),
    }

    def _note_target(note: Dict[str, Any]) -> float:
        tp = note.get("target_price")
        if tp is None:
            f0 = feats[0] if feats else {"hTop": 0.02, "vol": 1.0}
            bigger = [f for f in feats[1:]]
            hTop_big = (sum(f["hTop"] for f in bigger) / len(bigger)) if bigger else f0["hTop"]
            expected_up = ALPHA_H_TOP * f0["hTop"] + BETA_H_TOP * hTop_big
            return price * (1.0 + max(0.0, expected_up))
        return float(tp)

    def should_sell_note(note: Dict[str, Any]) -> bool:
        has_target = price >= _note_target(note)
        consensus_top = (top >= thresh_sell) and turning_down
        return has_target or consensus_top

    selected: List[Dict[str, Any]] = []
    for note in candidates:
        if len(selected) >= state["max_sells"]:
            break
        if should_sell_note(note):
            selected.append(note)

    if not selected:
        msg = (
            f"[HOLD][{window_name} {cfg['window_size']}] price=${price:.4f} Notes | "
            f"Open={open_count} | Closed={closed_count} | Next="
        )
        if next_target is not None:
            msg += f"${next_target:.4f}"
        else:
            msg += "None"
        addlog(msg, verbose_int=3, verbose_state=verbose)
        return []

    addlog(
        f"[MATURE][{window_name} {cfg['window_size']}] eligible={len(candidates)} sold={len(selected)} cap={state['max_sells']} "
        f"(top={top:.3f} ≥ {thresh_sell:.3f}, turn↓={turning_down})",
        verbose_int=1,
        verbose_state=verbose,
    )

    return selected
