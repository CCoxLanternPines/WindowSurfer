from __future__ import annotations

"""Evaluate whether to open a new buy position."""

from typing import Any, Dict, List, Tuple


def scale_buy_size(s: float, total_cap: float, cfg: Dict[str, float]) -> float:
    """Scale buy size based on bubble magnitude and portfolio."""
    min_bubble = cfg.get("buy_min_bubble", 0)
    max_bubble = cfg.get("buy_max_bubble", 0)
    if s < min_bubble:
        return 0.0
    s_clamped = min(max(s, min_bubble), max_bubble)
    frac = (s_clamped - min_bubble) / max(1e-9, (max_bubble - min_bubble))
    pct = cfg.get("min_note_size_pct", 0.0) + frac * (
        cfg.get("max_note_size_pct", 0.0) - cfg.get("min_note_size_pct", 0.0)
    )
    return total_cap * pct


def sell_target_from_bubble(entry_price: float, s: float, cfg: Dict[str, float]) -> float:
    """Compute sell target price from bubble size."""
    min_bubble = cfg.get("sell_min_bubble", 0)
    max_bubble = cfg.get("sell_max_bubble", 0)
    if s < min_bubble:
        return float("inf")
    s_clamped = min(max(s, min_bubble), max_bubble)
    frac = (s_clamped - min_bubble) / max(1e-9, (max_bubble - min_bubble))
    maturity = cfg.get("min_maturity", 0.0) + frac * (
        cfg.get("max_maturity", 0.0) - cfg.get("min_maturity", 0.0)
    )
    return entry_price * (1 + maturity)


def trend_multiplier_lerp(v: float, cfg: Dict[str, float]) -> float:
    """Linear interpolation of trend multiplier from normalized angle."""
    v = max(-1.0, min(1.0, float(v)))
    down = cfg.get("buy_mult_trend_down", 0.0)
    floor = cfg.get("buy_mult_trend_floor", 0.0)
    up = cfg.get("buy_mult_trend_up", 0.0)
    if v < 0:
        return down + (floor - down) * (v + 1)
    return floor + (up - floor) * v


def vol_multiplier(vol: float, cfg: Dict[str, float]) -> float:
    """Map rolling volatility into a buy multiplier."""
    min_b = cfg.get("buy_min_vol_bubble", 0.0)
    max_b = cfg.get("buy_max_vol_bubble", 0.0)
    v = min(max(vol, min_b), max_b)
    frac = (v - min_b) / max(1e-9, (max_b - min_b))
    return cfg.get("buy_mult_vol_min", 0.0) + frac * (
        cfg.get("buy_mult_vol_max", 0.0) - cfg.get("buy_mult_vol_min", 0.0)
    )


def evaluate_buy(
    idx: int,
    row: Dict[str, Any],
    pts: Dict[str, Dict[str, List[float]]],
    capital: float,
    open_notes: List[Dict[str, float]],
    cfg: Dict[str, float],
) -> Tuple[List[Dict[str, Any]], float, List[Dict[str, float]]]:
    """Evaluate a potential buy at the given candle index."""
    updated_notes = list(open_notes)
    trades: List[Dict[str, Any]] = []
    updated_capital = capital

    price = row["close"]

    if idx in pts["exhaustion_down"]["x"]:
        i = pts["exhaustion_down"]["x"].index(idx)
        bubble = pts["exhaustion_down"]["s"][i]
        total_cap = capital + sum(n["units"] * price for n in open_notes)
        trade_usd = scale_buy_size(bubble, total_cap, cfg)

        v = row["angle"]
        trend_mult = trend_multiplier_lerp(v, cfg)

        vol = row.get("volatility", 0)
        vol_mult = vol_multiplier(vol, cfg)

        trend_floor = cfg.get("buy_mult_trend_floor", 0.0)
        trade_usd *= max(trend_floor, trend_mult)
        trade_usd *= vol_mult

        if trade_usd > 0 and capital >= trade_usd:
            units = trade_usd / price
            updated_capital -= trade_usd
            sell_price = sell_target_from_bubble(price, bubble, cfg)
            updated_notes.append({"entry_price": price, "units": units, "sell_price": sell_price})
            trades.append({"idx": idx, "price": price, "side": "BUY", "usd": trade_usd})
            print(
                f"BUY @ idx={idx}, price={price:.2f}, angle_mult={trend_mult:.2f}, vol_mult={vol_mult:.2f}, target={sell_price:.2f}"
            )

    return trades, updated_capital, updated_notes

