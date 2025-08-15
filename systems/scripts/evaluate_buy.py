from __future__ import annotations

"""Position-based buy evaluation."""

from typing import Dict, Any

from systems.utils.addlog import addlog
import numpy as np

# ==== CONSENSUS TUNING (adjust here; no settings changes) =====================

# Use the same base series as the window, but compute multi-window stats here.
CONSENSUS_WINDOWS = [
    ("D",  "1d",  0.90),
    ("W",  "7d",  1.00),   # higher-frame veto power
    ("M", "30d",  0.80),
    ("Q", "90d",  0.60),
]

# Buy threshold / history for BottomScore
BOTTOM_Q        = 0.82
LOOKBACK_FOR_Q  = "150d"

# Slope smoothing on the smallest window
TURN_EMA_FRAC   = 0.15

# Higher-frame trend gate (block buys unless uptrend OR deep catch)
REQUIRE_HIGHER_TREND = True
HIGHER_TREND_MIN = 0.05      # weighted (W/M/Q) slope >= this OR
HIGHER_POS_MIN   = 0.00      # avg (W/M/Q) position >= this

# Deep-catch override (allow buys in downtrends only if truly deep)
ALLOW_DEEP_CATCH = True
DEEP_POS_D  = -0.75          # D position very low
DEEP_POS_WM = -0.50          # AND min(W,M) position very low

# Buy spacing so we don't ladder too fast on one leg
COOLDOWN_MOVE_FRAC = 0.22

# Target construction weights (handed to sell via note["target_price"])
ALPHA_H_TOP = 0.80
BETA_H_TOP  = 0.60


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
        if t + 1 >= bars:
            f = _window_features(series, t, bars)
            f["span_str"] = span_str
            f["bars"] = bars
            f["weight"] = w
            feats.append(f)
    if not feats:
        return 0.0, 0.0, [], False, False
    bottom = sum(f["weight"] * max(-f["pos"], 0.0) for f in feats)
    top = sum(f["weight"] * max(f["pos"], 0.0) for f in feats)
    turning_up = feats[0]["slope"] > 0.0
    turning_down = not turning_up
    return bottom, top, feats, turning_up, turning_down


def _precompute_consensus_scores(series, windows, turn_ema_frac):
    """Precompute bottom/top consensus scores for the full series."""
    N = len(series)
    bottom = np.zeros(N, dtype=float)
    top = np.zeros(N, dtype=float)
    for i in range(N):
        b, u, _f, _tu, _td = _consensus(series, i)
        bottom[i] = b
        top[i] = u
    return bottom, top


def _percentile(values, q: float) -> float:
    if values is None or len(values) == 0:
        return 0.0
    return float(np.quantile(np.array(values, dtype=float), q))


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

    # --- consensus + rolling threshold ---
    runtime_state = ctx.get("runtime_state", runtime_state)
    pre = runtime_state.setdefault("precomp", {})
    last_ts = int(series["timestamp"].iat[-1]) if "timestamp" in series else t
    key = f"{len(series)}::{last_ts}"
    bottom_arr = pre.get("bottom")
    top_arr = pre.get("top")
    lb = pre.get("lb_bars")
    if bottom_arr is None or top_arr is None:
        bottom_arr, top_arr = _precompute_consensus_scores(series, CONSENSUS_WINDOWS, TURN_EMA_FRAC)
        lb = _bars_for_span(series, LOOKBACK_FOR_Q)
        pre.update({"bottom": bottom_arr, "top": top_arr, "lb_bars": lb, "key": key})
    else:
        old_len = len(bottom_arr)
        if len(series) > old_len:
            new_b = []
            new_t = []
            for idx in range(old_len, len(series)):
                b, u, _f, _tu, _td = _consensus(series, idx)
                new_b.append(b)
                new_t.append(u)
            pre["bottom"] = bottom_arr = np.append(bottom_arr, new_b)
            pre["top"] = top_arr = np.append(top_arr, new_t)
            lb = _bars_for_span(series, LOOKBACK_FOR_Q)
            pre["lb_bars"] = lb
        elif pre.get("key") != key:
            bottom_arr, top_arr = _precompute_consensus_scores(series, CONSENSUS_WINDOWS, TURN_EMA_FRAC)
            lb = _bars_for_span(series, LOOKBACK_FOR_Q)
            pre.update({"bottom": bottom_arr, "top": top_arr, "lb_bars": lb})
        pre["key"] = key

    lb = pre["lb_bars"]
    min_warm = max(lb, _bars_for_span(series, "30d")) // 4
    if t < min_warm:
        return False

    bottom = float(pre["bottom"][t])
    top = float(pre["top"][t])
    start = max(0, t - lb + 1)
    thresh_buy = _percentile(pre["bottom"][start : t + 1], BOTTOM_Q)
    _, _, feats, turning_up, _ = _consensus(series, t)

    # --- spacing (distance since last buy in this window) ---
    last_buy_idx_key = f"last_buy_idx::{window_name}"
    last_buy_idx = runtime_state.get(last_buy_idx_key)
    ok_move = True
    if last_buy_idx is not None and feats:
        f0 = feats[0]
        prev_px = float(series["close"].iat[last_buy_idx])
        move = abs(float(candle["close"]) - prev_px)
        ok_move = move >= (COOLDOWN_MOVE_FRAC * max(f0["width"], 1e-9))

    # --- higher-frame trend gate + deep-catch override ---
    higher = feats[1:] if len(feats) > 1 else []
    uptrend_ok, deep_ok = True, False
    if REQUIRE_HIGHER_TREND and higher:
        wsum = sum(f["weight"] for f in higher) or 1.0
        weighted_slope = sum(f["weight"]*f["slope"] for f in higher)/wsum
        avg_pos        = sum(f["weight"]*f["pos"]   for f in higher)/wsum
        uptrend_ok = (weighted_slope >= HIGHER_TREND_MIN) or (avg_pos >= HIGHER_POS_MIN)
        f0 = feats[0]
        wm = [f for f in higher if f.get("span_str") in ("7d","30d")]
        min_wm_pos = min((f["pos"] for f in wm), default=0.0)
        deep_ok = ALLOW_DEEP_CATCH and (f0["pos"] <= DEEP_POS_D) and (min_wm_pos <= DEEP_POS_WM)

    _should_buy_gate = (bottom >= thresh_buy) and turning_up and ok_move and (uptrend_ok or deep_ok)
    if not _should_buy_gate:
        return False

    addlog(
        f"[BUY?][{window_name} {cfg['window_size']}] bottom={bottom:.3f} ≥ {thresh_buy:.3f} turn↑={turning_up} uptrend={uptrend_ok} deep={deep_ok} move={ok_move}",
        verbose_int=3,
        verbose_state=ctx.get("verbose"),
    )

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

    price_now = float(candle["close"])
    if feats:
        f0 = feats[0]
        bigger = [f for f in feats[1:]]
        hTop_big = (sum(f["hTop"] for f in bigger)/len(bigger)) if bigger else f0["hTop"]
        expected_up = ALPHA_H_TOP * f0["hTop"] + BETA_H_TOP * hTop_big
        target_price = price_now * (1.0 + max(0.0, expected_up))
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

    runtime_state[last_buy_idx_key] = t

    return result
