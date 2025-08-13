"""Simple self-funded jackpot DCA strategy."""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import pandas as pd

from .candle_utils import compute_ath_atl


# ---------------------------------------------------------------------------
# State helpers
# ---------------------------------------------------------------------------


def init_state(tag: str, ledger, settings: Dict, now_ts: int) -> Dict:
    """Return initial state for a jackpot strategy instance."""
    return {
        "last_drip_ts": None,
        "total_contributed_usd": 0.0,
        "inventory_qty": 0.0,
        "avg_entry_price": None,
        "ath_price": None,
        "atl_price": None,
        "realized_pnl": 0.0,
    }


# ---------------------------------------------------------------------------
# Reference levels
# ---------------------------------------------------------------------------


def update_reference_levels(
    candles_df: pd.DataFrame,
    scope: str,
    full_history_df: Optional[pd.DataFrame] = None,
) -> Tuple[float | None, float | None]:
    """Return the ATH and ATL depending on ``scope``.

    Parameters
    ----------
    candles_df:
        Dataframe of the current in-memory dataset.
    scope:
        Either ``"dataset"`` or ``"full_history"``. When ``"full_history"`` is
        requested ``full_history_df`` is used if provided; otherwise the
        function falls back to ``candles_df``.
    full_history_df:
        Optional dataframe representing the full history of the asset.
    """
    if scope == "full_history" and full_history_df is not None:
        return compute_ath_atl(full_history_df)
    return compute_ath_atl(candles_df)


# ---------------------------------------------------------------------------
# Core math utilities
# ---------------------------------------------------------------------------


def eligibility(price: float, ath: float, atl: float, start_frac: float) -> Tuple[bool, float]:
    """Determine if current ``price`` is eligible for dripping and return
    position ``pos`` in [0,1]."""
    if ath is None or atl is None or ath <= 0 or atl >= ath:
        return False, 0.0
    start_line = ath * start_frac
    if price > start_line:
        return False, 0.0
    p_clamped = max(min(price, start_line), atl)
    pos = (start_line - p_clamped) / (start_line - atl) if start_line > atl else 0.0
    return True, pos


def compute_drip_usd(
    cfg: Dict,
    state: Dict,
    realized_pnl_available: float,
    pos: float,
) -> float:
    base = cfg.get("base_drip_usd", 0.0)
    mult = 1.0 + pos * (cfg.get("multiplier_floor", 1.0) - 1.0)
    desired = base * mult

    if cfg.get("invest_only_from_realized_profits", True):
        contributed = state.get("total_contributed_usd", 0.0)
        budget = max(0.0, realized_pnl_available - contributed)
        desired = min(desired * cfg.get("profit_invest_fraction", 1.0), budget)

    fee_bps = cfg.get("fee_bps", 0)
    fee_buffer = desired * fee_bps / 10000
    min_order = cfg.get("min_order_usd", 0.0) + fee_buffer
    if desired < min_order:
        return 0.0
    return desired


def maybe_drip(now_ts: int, last_drip_ts: Optional[int], period_hours: float) -> bool:
    if last_drip_ts is None:
        return True
    return (now_ts - last_drip_ts) >= period_hours * 3600


# ---------------------------------------------------------------------------
# Signal evaluation
# ---------------------------------------------------------------------------


def evaluate_buy(
    state: Dict,
    price: float,
    now_ts: int,
    realized_pnl_available: float,
    cfg: Dict,
) -> Optional[Dict]:
    ath = state.get("ath_price")
    atl = state.get("atl_price")
    eligible, pos = eligibility(price, ath, atl, cfg.get("start_level_frac", 0.5))
    if not eligible:
        return None
    if not maybe_drip(now_ts, state.get("last_drip_ts"), cfg.get("drip_period_hours", 0)):
        return None
    usd = compute_drip_usd(cfg, state, realized_pnl_available, pos)
    if usd <= 0:
        return None
    qty = usd / price if price > 0 else 0.0
    return {"type": "BUY", "qty": qty, "usd": usd, "reason": f"jackpot_drip pos={pos:.2f}"}


def evaluate_sell(state: Dict, price: float, cfg: Dict) -> Optional[Dict]:
    ath = state.get("ath_price")
    if ath is None or state.get("inventory_qty", 0.0) <= 0:
        return None
    take_frac = cfg.get("take_profit_frac", 0.75)
    exit_new_ath = cfg.get("exit_on_new_ath", True)
    if price >= take_frac * ath or (exit_new_ath and price > ath):
        return {
            "type": "SELL_ALL",
            "qty": state.get("inventory_qty", 0.0),
            "reason": "jackpot_take_profit",
        }
    return None


# ---------------------------------------------------------------------------
# Fill application
# ---------------------------------------------------------------------------


def apply_fills(state: Dict, fills: list[Dict], ledger=None) -> None:
    for fill in fills:
        ftype = fill.get("type")
        if ftype == "BUY":
            qty = float(fill.get("qty", 0.0))
            usd = float(fill.get("usd", 0.0))
            price = float(fill.get("price", 0.0))
            inv = state.get("inventory_qty", 0.0)
            total_cost = (state.get("avg_entry_price", 0.0) or 0.0) * inv
            total_cost += price * qty
            inv += qty
            state["inventory_qty"] = inv
            state["avg_entry_price"] = total_cost / inv if inv > 0 else None
            state["total_contributed_usd"] = state.get("total_contributed_usd", 0.0) + usd
            state["last_drip_ts"] = fill.get("timestamp")
            if ledger is not None:
                ledger.record_trade(
                    {
                        "strategy": "jackpot",
                        "event": "buy",
                        "qty": qty,
                        "usd": usd,
                        "price": price,
                        "timestamp": fill.get("timestamp"),
                    }
                )
        elif ftype == "SELL_ALL":
            qty = float(fill.get("qty", 0.0))
            price = float(fill.get("price", 0.0))
            avg = state.get("avg_entry_price", 0.0) or 0.0
            pnl = (price - avg) * qty
            state["realized_pnl"] = state.get("realized_pnl", 0.0) + pnl
            state["inventory_qty"] = 0.0
            state["avg_entry_price"] = None
            if ledger is not None:
                ledger.record_trade(
                    {
                        "strategy": "jackpot",
                        "event": "sell_all",
                        "qty": qty,
                        "usd": price * qty,
                        "price": price,
                        "cost": avg * qty,
                        "timestamp": fill.get("timestamp"),
                    }
                )
            # total_contributed_usd intentionally not reset
