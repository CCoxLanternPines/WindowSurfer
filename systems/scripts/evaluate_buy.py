from __future__ import annotations

"""Pressure-based buy evaluator with optional rich logging and strategy knobs."""

from typing import Any, Dict, Tuple
import math
import os
import pandas as pd

from systems.utils.addlog import addlog


# --- tuning constants -----------------------------------------------------

# Window spans and weights used for pressure computation
PRESSURE_WINDOWS = [
    ("1d", 0.15),
    ("3d", 0.15),
    ("1w", 0.15),
    ("2w", 0.15),
    ("1m", 0.20),
    ("3m", 0.20),
]

MIN_GAP_BARS = 12

NORM_N = 10.0
NORM_V = 5_000.0
NORM_A = 100.0
NORM_R = 0.10

BP_SIG_CENTER = 0.40
BP_SIG_WIDTH = 0.12


# ---------------------------------------------------------------------------


def _clamp01(x: float) -> float:
    return max(0.0, min(1.0, x))


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def _span_to_bars(series: pd.DataFrame, span: str) -> int:
    if not span:
        return 1
    try:
        value = int(span[:-1])
    except ValueError:
        return 1
    unit = span[-1].lower()
    if "timestamp" in series.columns and len(series) >= 2:
        step = int(series.iloc[1]["timestamp"]) - int(series.iloc[0]["timestamp"])
        if step <= 0:
            step = 3600
    else:
        step = 3600
    if unit == "m":
        factor = 30 * 24 * 3600
    elif unit == "w":
        factor = 7 * 24 * 3600
    elif unit == "d":
        factor = 24 * 3600
    elif unit == "h":
        factor = 3600
    else:
        factor = 0
    bars = int(round((value * factor) / step))
    return max(1, bars)


def _window_scores(series: pd.DataFrame, t: int) -> Dict[str, Dict[str, float]]:
    scores: Dict[str, Dict[str, float]] = {}
    close_now = float(series.iloc[t]["close"])
    for span, _ in PRESSURE_WINDOWS:
        bars = _span_to_bars(series, span)
        start = max(0, t - bars + 1)
        window = series.iloc[start : t + 1]
        low = float(window["close"].min())
        high = float(window["close"].max())
        width = max(high - low, 1e-9)
        depth = (high - close_now) / width
        height = (close_now - low) / width
        scores[span] = {"depth": depth, "height": height}
    return scores


def _compute_pressures(
    series: pd.DataFrame, t: int
) -> Tuple[float, float, Dict[str, Dict[str, float]]]:
    scores = _window_scores(series, t)
    bp = 0.0
    sp = 0.0
    for span, weight in PRESSURE_WINDOWS:
        sc = scores.get(span, {})
        bp += weight * sc.get("depth", 0.0)
        sp += weight * sc.get("height", 0.0)
    return bp, sp, scores


def _compute_sp_load(
    notes, price_now: float, t: int
) -> Tuple[float, int, float, float, float, float, float, float]:
    """Return sell-pressure load and its components."""

    N = len(notes)
    val = 0.0
    ages = []
    rois = []
    for n in notes:
        qty = n.get("entry_amount")
        if qty is not None:
            val += float(qty) * price_now
        else:
            val += float(n.get("entry_usd", 0.0))
        created = n.get("created_idx", t)
        ages.append(t - created)
        buy = n.get("entry_price", 0.0)
        if buy:
            rois.append(max((price_now - buy) / buy, 0.0))
    avg_age = sum(ages) / N if N else 0.0
    avg_roi_pos = sum(rois) / N if N else 0.0
    N_norm = min(1.0, N / NORM_N)
    V_norm = min(1.0, val / NORM_V)
    A_norm = min(1.0, avg_age / NORM_A)
    R_norm = min(1.0, avg_roi_pos / NORM_R)
    load = 0.35 * N_norm + 0.35 * V_norm + 0.15 * A_norm + 0.15 * R_norm
    return _clamp01(load), N, val, avg_age, N_norm, V_norm, A_norm, R_norm


def _compute_regime(series: pd.DataFrame, t: int) -> Tuple[str, float, float, float, float]:
    def _slope(bars: int) -> float:
        idx = max(0, t - bars)
        past = float(series.iloc[idx]["close"])
        now = float(series.iloc[t]["close"])
        return (now - past) / max(past, 1e-9)

    b1 = _span_to_bars(series, "1d")
    b3 = _span_to_bars(series, "3d")
    b7 = _span_to_bars(series, "1w")
    s1 = _slope(b1)
    s3 = _slope(b3)
    s7 = _slope(b7)

    start = max(0, t - b7 + 1)
    window = series.iloc[start : t + 1]["close"].pct_change().dropna()
    vol = float(window.std()) if not window.empty else 0.0

    if s1 > 0 and s3 > 0 and s7 > 0:
        regime = "Bull"
    elif s1 < 0 and s3 < 0 and s7 < 0:
        regime = "Bear"
    else:
        regime = "Chop"
    return regime, s1, s3, s7, vol


def _dd_buffer(series: pd.DataFrame, runtime_state: Dict[str, Any], t: int) -> Tuple[bool, float, float, int]:
    """Return (throttle, equity, dd, days_left)."""

    env = os.environ
    thresh = float(env.get("WS_BUFFER_DD_THRESH", "0"))
    days = int(env.get("WS_BUFFER_DAYS", "0"))
    info = runtime_state.setdefault("_dd_buffer", {"until": -1})
    bars_day = _span_to_bars(series, "1d")
    days_left = max(0, (info["until"] - t) // bars_day)

    eq = float(runtime_state.get("equity_usd", 0.0))
    peak = float(runtime_state.get("equity_peak", eq))
    dd = (eq - peak) / peak if peak else 0.0

    throttle = days_left > 0
    if not throttle and dd < -thresh and days > 0:
        throttle = True
        info["until"] = t + days * bars_day
        days_left = days

    return throttle, eq, dd, days_left


def evaluate_buy(
    ctx: Dict[str, Any],
    t: int,
    series: pd.DataFrame,
    *,
    window_name: str,
    cfg: Dict[str, Any],
    runtime_state: Dict[str, Any],
):
    """Return sizing and metadata for a buy signal in ``window_name``."""

    env = os.environ
    LOG = env.get("WS_LOG_DECISIONS") == "1"
    PRESSURE_MODEL = env.get("WS_PRESSURE_MODEL") == "1"
    REGIME_LOG = env.get("WS_REGIME_LOG") == "1"
    REGIME_BUY_MULT = env.get("WS_REGIME_BUY_MULT") == "1"
    BUFFER_DD = env.get("WS_BUFFER_DD") == "1"
    REGIME_CAP_ALLOC = env.get("WS_REGIME_CAP_ALLOC") == "1"
    VETO_HIGHHTF = env.get("WS_VETO_HIGHHTF") == "1"

    verbose = runtime_state.get("verbose", 0)

    bp_price, sp_price, scores = _compute_pressures(series, t)
    if not PRESSURE_MODEL:
        bp_price = scores.get("1d", {}).get("depth", 0.0)
        sp_price = scores.get("1d", {}).get("height", 0.0)

    close_now = float(series.iloc[t]["close"])

    ledger = ctx.get("ledger")
    notes = ledger.get_open_notes() if ledger is not None else []
    sp_load, N_notes, val_notes, avg_age, N_norm, V_norm, A_norm, R_norm = _compute_sp_load(
        notes, close_now, t
    )
    sp_total = _clamp01(0.6 * sp_price + 0.4 * sp_load)

    regime, s1, s3, s7, vol = "None", 0.0, 0.0, 0.0, 0.0
    reg_mult = 1.0
    if REGIME_LOG or REGIME_BUY_MULT or REGIME_CAP_ALLOC:
        regime, s1, s3, s7, vol = _compute_regime(series, t)
        mult_map = {"Bull": 1.2, "Bear": 0.8, "Chop": 1.0}
        reg_mult = mult_map.get(regime, 1.0)

    if REGIME_BUY_MULT:
        bp_price *= reg_mult

    m_buy = _sigmoid((bp_price - BP_SIG_CENTER) / BP_SIG_WIDTH)

    capital = float(runtime_state.get("capital", 0.0))
    if REGIME_CAP_ALLOC:
        capital *= reg_mult

    limits = runtime_state.get("limits", {})
    min_sz = float(limits.get("min_note_size", 0.0))
    max_sz = float(limits.get("max_note_usdt", float("inf")))

    base = float(cfg.get("investment_fraction", 0.0))
    size_usd = capital * base * m_buy
    raw = size_usd
    size_usd = min(size_usd, capital, max_sz)
    if raw != size_usd and LOG:
        addlog(
            f"[CLAMP] size=${raw:.2f} → ${size_usd:.2f} (cap=${capital:.2f}, max=${max_sz:.2f})",
            verbose_int=2,
            verbose_state=verbose,
        )

    cooldown_ok = True
    last_key = f"last_buy_idx::{window_name}"
    last_idx = int(runtime_state.get(last_key, -1))
    if last_idx >= 0:
        cooldown_ok = (t - last_idx) >= MIN_GAP_BARS

    veto = False
    if VETO_HIGHHTF:
        htf_height = max(
            scores.get("1m", {}).get("height", 0.0),
            scores.get("3m", {}).get("height", 0.0),
        )
        veto = htf_height > 0.75

    throttle = False
    eq, dd, dd_days = 0.0, 0.0, 0
    if BUFFER_DD:
        throttle, eq, dd, dd_days = _dd_buffer(series, runtime_state, t)

    decision: Dict[str, Any] | bool = False
    reason = ""
    if not cooldown_ok:
        reason = "cooldown"
    elif veto:
        reason = "veto"
    elif throttle:
        reason = "dd_throttle"
    elif size_usd < min_sz or size_usd <= 0.0:
        reason = "size"
    else:
        decision = {
            "size_usd": size_usd,
            "window_name": window_name,
            "window_size": cfg["window_size"],
            "created_idx": t,
        }
        if "timestamp" in series.columns:
            decision["created_ts"] = int(series.iloc[t]["timestamp"])
        runtime_state[last_key] = t
        reason = "buy"

    if LOG and REGIME_LOG:
        addlog(
            f"[REGIME] t={t} slopes(1d,3d,1w)=({s1:+.2f},{s3:+.2f},{s7:+.2f}) vol_1w={vol:.3f} → {regime}",
            verbose_int=1,
            verbose_state=verbose,
        )

    if LOG and BUFFER_DD:
        addlog(
            f"[DD_BUFFER] 30d_eq={eq:.2f} dd={dd:.3f} → throttle={'on' if throttle else 'off'} days_left={dd_days}",
            verbose_int=1,
            verbose_state=verbose,
        )

    if LOG:
        depths = [scores.get(span, {}).get("depth", 0.0) for span, _ in PRESSURE_WINDOWS]
        bp_w = ",".join(
            f"{span}:{val:.2f}" for (span, _), val in zip(PRESSURE_WINDOWS, depths)
        )
        addlog(
            f"[BUY?][{window_name} {cfg['window_size']}] t={t} px={close_now:.4f}",
            verbose_int=1,
            verbose_state=verbose,
        )
        addlog(
            f"BP={bp_price:.3f}|BP_w=[{bp_w}]",
            verbose_int=1,
            verbose_state=verbose,
        )
        addlog(
            f"reg={regime} mult={m_buy:.2f} cool_ok={cooldown_ok} veto={veto}",
            verbose_int=1,
            verbose_state=verbose,
        )
        addlog(
            f"size_raw=${raw:.2f} clamp(${min_sz:.2f},{max_sz:.2f})→${size_usd:.2f} cap=${capital:.2f}",
            verbose_int=1,
            verbose_state=verbose,
        )
        addlog(
            f"decision={'BUY' if decision else 'SKIP'} reason={reason}",
            verbose_int=1,
            verbose_state=verbose,
        )

        addlog(
            f"[SP_LOAD] N={N_norm:.2f} V={V_norm:.2f} A={A_norm:.2f} R={R_norm:.2f}",
            verbose_int=2,
            verbose_state=verbose,
        )

    return decision


