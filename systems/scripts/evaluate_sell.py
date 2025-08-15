from __future__ import annotations

"""Price-target-based sell evaluation."""

from typing import Any, Dict, List

from systems.utils.addlog import addlog

# ==== DOWNTREND SELL ACCELERATION (adjust here; no settings changes) ==========

CONSENSUS_WINDOWS = [
    ("D",  "1d",  0.80),
    ("W",  "7d",  0.90),
    ("M", "30d",  0.70),
    ("Q", "90d",  0.50),
]
TURN_EMA_FRAC     = 0.20


TOP_Q             = 0.98     # normal sell threshold (TopScore percentile)
LOOKBACK_FOR_Q    = "180d"

# When higher frames are down, allow faster exits:
DOWNTREND_TOP_Q   = 0.70     # lower TopScore percentile during downtrends
HIGHER_TREND_MIN  = 0.03     # consider downtrend if weighted slope <= -this
HIGHER_POS_MAX    = -0.10    # or avg position <= this

# Optional bounce-sell (profit taking on relief pops):
BOUNCE_SELL_MIN_AGE = 6      # bars since buy
MIN_ROI_DOWN        = 0.003  # ≥0.3% ROI to bounce-sell
# ============================================================================




def _parse_span_seconds(s: str) -> int:
    s = s.strip()
    n = int("".join(ch for ch in s if ch.isdigit()) or "1")
    u = "".join(ch for ch in s if ch.isalpha()).lower() or "h"
    return n * {"m":60,"h":3600,"d":86400,"w":604800}[u]


def _detect_step_seconds(series) -> int:
    import numpy as np
    if "timestamp" not in series.columns or len(series) < 3: return 3600
    ts = series["timestamp"].to_numpy()
    diffs = np.diff(ts[-200:]) if ts.size>200 else np.diff(ts)
    med = int(np.median(diffs)) if diffs.size else 3600
    return med or 3600


def _bars_for_span(series, span_str: str) -> int:
    return max(2, _parse_span_seconds(span_str)//_detect_step_seconds(series))


def _window_features(series, t: int, bars: int) -> dict:
    import numpy as np, pandas as pd
    lo = max(0, t - bars + 1); w = series.iloc[lo:t+1]
    px = w["close"].to_numpy(dtype=float)
    if px.size < 2:
        now = float(series["close"].iat[t]); z=1e-9
        return dict(low=now, high=now, mid=now, width=z, pos=0.0, slope=0.0, vol=0.0, hTop=0.0, hBot=0.0)
    low, high = float(px.min()), float(px.max()); width = max(high-low, 1e-9)
    mid = (low+high)/2.0; pos = max(-1.0, min(1.0, (px[-1]-mid)/(width/2.0)))
    span = max(2, int(bars * TURN_EMA_FRAC))
    slope = float(
        (pd.Series(px).diff().fillna(0.0)
           .ewm(span=span, adjust=False).mean().iloc[-1])
    ) / width
    now = float(px[-1])
    hTop = max(high-now,0.0)/max(now,1e-9)
    hBot = max(now-low,0.0)/max(now,1e-9)
    return dict(low=low, high=high, mid=mid, width=width, pos=pos, slope=slope, hTop=hTop, hBot=hBot)


def _consensus(series, t: int):
    feats = []
    for _, span_str, w in CONSENSUS_WINDOWS:
        bars = _bars_for_span(series, span_str)
        if t+1 >= bars:
            f=_window_features(series,t,bars); f["span_str"]=span_str; f["bars"]=bars; f["weight"]=w; feats.append(f)
    if not feats: return 0.0, 0.0, [], False, False
    bottom = sum(f["weight"]*max(-f["pos"],0.0) for f in feats)
    top    = sum(f["weight"]*max( f["pos"],0.0) for f in feats)
    turning_up   = feats[0]["slope"] > 0.0
    turning_down = not turning_up
    return bottom, top, feats, turning_up, turning_down


def _score_cache(rs: dict) -> dict:
    key="consensus_cache_sell"
    if key not in rs: rs[key]={"bottom":[], "top":[]}
    return rs[key]


def _bars_for_lookback(series, lookback_str: str) -> int:
    return _bars_for_span(series, lookback_str)


def _percentile(values, q: float) -> float:
    import numpy as np
    if not values: return 0.0
    return float(np.quantile(np.array(values, dtype=float), q))


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

    # --- consensus at current bar ---
    runtime_state = ctx.get("runtime_state", runtime_state if runtime_state is not None else {})
    bottom, top, feats, _turn_up, turning_down = _consensus(series, t)
    cache = _score_cache(runtime_state)
    cache["top"].append(top)

    lb = _bars_for_lookback(series, LOOKBACK_FOR_Q)
    thresh_normal = _percentile(cache["top"][-lb:], TOP_Q) if lb>0 else _percentile(cache["top"], TOP_Q)

    # downtrend flag from higher frames
    higher = feats[1:] if len(feats)>1 else []
    is_downtrend = False
    if higher:
        wsum = sum(f["weight"] for f in higher) or 1.0
        w_slope = sum(f["weight"]*f["slope"] for f in higher)/wsum
        w_pos   = sum(f["weight"]*f["pos"]   for f in higher)/wsum
        is_downtrend = (w_slope <= -HIGHER_TREND_MIN) or (w_pos <= HIGHER_POS_MAX)

    thresh_down = _percentile(cache["top"][-lb:], DOWNTREND_TOP_Q) if lb>0 else _percentile(cache["top"], DOWNTREND_TOP_Q)

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
            return price * 1.02
        return float(tp)

    def should_sell_note(note: Dict[str, Any]) -> bool:
        has_target = price >= _note_target(note)
        consensus_top = (top >= thresh_normal) and turning_down
        return has_target or consensus_top

    selected: List[Dict[str, Any]] = []
    for note in candidates:
        if len(selected) >= state["max_sells"]:
            break
        if should_sell_note(note):
            selected.append(note)

    def _roi(note):
        ep = float(note.get("entry_price", price))
        return (price/ep) - 1.0

    def _age(note):
        ci = note.get("created_idx")
        return (t - int(ci)) if ci is not None else 1_000_000

    if is_downtrend:
        accel = []
        for note in candidates:
            if note in selected:
                continue
            ok_consensus = (top >= thresh_down) and turning_down
            ok_bounce    = (_age(note) >= BOUNCE_SELL_MIN_AGE) and (_roi(note) >= MIN_ROI_DOWN)
            has_target   = price >= float(note.get("target_price", float("inf")))
            if ok_consensus or ok_bounce or has_target:
                accel.append(note)
        # merge, respect per-candle cap
        cap = state.get("max_sells", len(candidates))
        merged = selected + [n for n in accel if n not in selected]
        selected = merged[:cap]

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

    thr = thresh_down if is_downtrend else thresh_normal
    addlog(
        f"[MATURE][{window_name} {cfg['window_size']}] eligible={len(candidates)} "
        f"sold={len(selected)} cap={state['max_sells']} "
        f"(top={top:.3f} thr={thr:.3f} downtrend={is_downtrend} turn↓={turning_down})",
        verbose_int=1,
        verbose_state=verbose,
    )

    return selected

