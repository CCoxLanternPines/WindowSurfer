from __future__ import annotations

"""Evaluate whether to open a new buy position."""

from typing import Any, Dict, List, Tuple


# Buy scaling constants
BUY_MIN_BUBBLE    = 100
BUY_MAX_BUBBLE    = 500
MIN_NOTE_SIZE_PCT = 0.03    # 1% of portfolio
MAX_NOTE_SIZE_PCT = 0.25    # 5% of portfolio

# Sell scaling (baked into note at buy time)
SELL_MIN_BUBBLE   = 150
SELL_MAX_BUBBLE   = 800
MIN_MATURITY      = 0.05    # 0% gain (sell at entry)
MAX_MATURITY      = .25     # 100% gain (2x entry)

# Volatility buy scaling
BUY_MIN_VOL_BUBBLE = 0
BUY_MAX_VOL_BUBBLE = .01
BUY_MULT_VOL_MIN   = 2.5
BUY_MULT_VOL_MAX   = 0

# Trend multipliers
BUY_MULT_TREND_UP   = 1   # strong up-trend multiplier (cap at +1 normalized)
BUY_MULT_TREND_FLOOR = .25  # keep 0 so flat maps to 0, no forced minimum
BUY_MULT_TREND_DOWN = 0   # strong down-trend multiplier (cap at -1 normalized)


def scale_buy_size(s: float, total_cap: float) -> float:
    if s < BUY_MIN_BUBBLE:
        return 0.0
    s_clamped = min(max(s, BUY_MIN_BUBBLE), BUY_MAX_BUBBLE)
    frac = (s_clamped - BUY_MIN_BUBBLE) / (BUY_MAX_BUBBLE - BUY_MIN_BUBBLE)
    pct = MIN_NOTE_SIZE_PCT + frac * (MAX_NOTE_SIZE_PCT - MIN_NOTE_SIZE_PCT)
    return total_cap * pct


def sell_target_from_bubble(entry_price: float, s: float) -> float:
    if s < SELL_MIN_BUBBLE:
        return float("inf")  # effectively never sell
    s_clamped = min(max(s, SELL_MIN_BUBBLE), SELL_MAX_BUBBLE)
    frac = (s_clamped - SELL_MIN_BUBBLE) / (SELL_MAX_BUBBLE - SELL_MIN_BUBBLE)
    maturity = MIN_MATURITY + frac * (MAX_MATURITY - MIN_MATURITY)
    return entry_price * (1 + maturity)


def trend_multiplier_lerp(v: float) -> float:
    """Linear interpolation of trend multiplier from normalized angle."""
    v = max(-1.0, min(1.0, float(v)))
    if v < 0:
        return BUY_MULT_TREND_DOWN + (BUY_MULT_TREND_FLOOR - BUY_MULT_TREND_DOWN) * (v + 1)
    return BUY_MULT_TREND_FLOOR + (BUY_MULT_TREND_UP - BUY_MULT_TREND_FLOOR) * v


def vol_multiplier(vol: float) -> float:
    """Map rolling volatility into a buy multiplier."""
    v = min(max(vol, BUY_MIN_VOL_BUBBLE), BUY_MAX_VOL_BUBBLE)
    frac = (v - BUY_MIN_VOL_BUBBLE) / max(1e-9, (BUY_MAX_VOL_BUBBLE - BUY_MIN_VOL_BUBBLE))
    return BUY_MULT_VOL_MIN + frac * (BUY_MULT_VOL_MAX - BUY_MULT_VOL_MIN)


def evaluate_buy(
    idx: int,
    row: Dict[str, Any],
    pts: Dict[str, Dict[str, List[float]]],
    capital: float,
    open_notes: List[Dict[str, float]],
) -> Tuple[Dict[str, Any] | None, float, List[Dict[str, float]]]:
    """Evaluate a potential buy at the given candle index.

    Returns a tuple of (trade_dict_or_None, new_capital, new_open_notes).
    """
    updated_notes = list(open_notes)
    maybe_trade: Dict[str, Any] | None = None
    updated_capital = capital

    price = row["close"]

    if idx in pts["exhaustion_down"]["x"]:
        i = pts["exhaustion_down"]["x"].index(idx)
        bubble = pts["exhaustion_down"]["s"][i]
        total_cap = capital + sum(n["units"] * price for n in open_notes)
        trade_usd = scale_buy_size(bubble, total_cap)

        v = row["angle"]
        trend_mult = trend_multiplier_lerp(v)

        vol = row.get("volatility", 0)
        vol_mult = vol_multiplier(vol)

        trade_usd *= max(BUY_MULT_TREND_FLOOR, trend_mult)
        trade_usd *= vol_mult

        if trade_usd > 0 and capital >= trade_usd:
            units = trade_usd / price
            updated_capital -= trade_usd
            sell_price = sell_target_from_bubble(price, bubble)
            updated_notes.append({"entry_price": price, "units": units, "sell_price": sell_price})
            maybe_trade = {
                "idx": idx,
                "price": price,
                "side": "BUY",
                "usd": trade_usd,
                "units": units,
                "target": sell_price,
            }
            print(
                f"BUY @ idx={idx}, price={price:.2f}, angle_mult={trend_mult:.2f}, vol_mult={vol_mult:.2f}, target={sell_price:.2f}"
            )

    return maybe_trade, updated_capital, updated_notes
