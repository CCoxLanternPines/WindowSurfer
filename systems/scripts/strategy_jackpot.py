from __future__ import annotations

"""Simplified jackpot strategy scaffold."""

from dataclasses import dataclass
from typing import Dict, List

from systems.utils.logger import jp_info, jp_debug, jp_trace
from systems.utils.telegram import notify_telegram
from systems.utils.timeparse import parse_duration_to_hours


@dataclass
class JackpotConfig:
    enabled: bool
    start_level_frac: float
    take_profit_frac: float
    drip_period_hours: str
    base_drip_usd: float
    multiplier_max: float


def init_state(tag: str, ledger, settings: Dict, now_ts: int) -> dict:
    """Initialise jackpot state structure."""
    cfg = JackpotConfig(**settings)
    period_h = parse_duration_to_hours(cfg.drip_period_hours)
    jp_info(
        (
            "[JACKPOT][INIT] enabled={enabled} start={start:.0%} take={take:.0%} "
            "period={period}h base=${base:.2f} mult_max={mult:.2f}"
        ).format(
            enabled=cfg.enabled,
            start=cfg.start_level_frac,
            take=cfg.take_profit_frac,
            period=period_h,
            base=cfg.base_drip_usd,
            mult=cfg.multiplier_max,
        )
    )
    return {
        "tag": tag,
        "inventory_qty": 0.0,
        "avg_entry_price": 0.0,
        "total_contributed_usd": 0.0,
        "last_drip_ts": 0,
        "ATH": 0.0,
        "ATL": 0.0,
        "verbose": 1,
    }


def update_reference_levels(candles_df) -> tuple[float, float]:
    """Return (ATH, ATL) from candle dataframe."""
    if candles_df.empty:
        return 0.0, 0.0
    ath = float(candles_df["high"].max())
    atl = float(candles_df["low"].min())
    return ath, atl


def evaluate_jackpot(state: dict, price: float, now_ts: int, ledger_view, cfg: Dict) -> List[dict]:
    """Evaluate jackpot signals. Returns list of signal dicts."""
    signals: List[dict] = []
    if not cfg.get("enabled"):
        return signals

    ath = state.get("ATH", 0.0)
    atl = state.get("ATL", 0.0)
    start_line = ath * cfg.get("start_level_frac", 0.5)
    if start_line == atl:
        jp_debug("[JACKPOT][SKIP] reason=degenerate_levels S==ATL")
        return signals

    take_line = ath * cfg.get("take_profit_frac", 0.75)
    if price >= take_line:
        qty = state.get("inventory_qty", 0.0)
        if qty > 0:
            signals.append({"action": "sell_all", "qty": qty})
        else:
            jp_debug("[JACKPOT][EXIT] gate_open but no_inventory")
        return signals

    if price > start_line:
        return signals

    period_h = parse_duration_to_hours(cfg.get("drip_period_hours", 0))
    last = state.get("last_drip_ts", 0)
    last_age_h = (now_ts - last) / 3600 if last else period_h
    p_clamped = max(min(price, start_line), atl)
    pos = (start_line - p_clamped) / (start_line - atl) if start_line != atl else 0.0
    mult = 1.0 + pos * (cfg.get("multiplier_max", 1.0) - 1.0)
    desired = cfg.get("base_drip_usd", 0.0) * mult

    closed = getattr(ledger_view, "get_closed_notes", lambda: [])()
    realized = sum(n.get("gain", 0.0) for n in closed)
    budget_left = realized - state.get("total_contributed_usd", 0.0)

    jp_debug(
        (
            "[JACKPOT][ELIGIBLE] price=${p:.4f} ≤ start=${s:.4f} pos={pos:.2f} "
            "mult={m:.2f} desired=${d:.2f} budget=${b:.2f}"
        ).format(p=price, s=start_line, pos=pos, m=mult, d=desired, b=budget_left)
    )
    jp_debug(f"[JACKPOT][CADENCE] last={last_age_h:.1f}h ≥ period={period_h}h")

    if budget_left <= 0:
        jp_debug(
            (
                "[JACKPOT][SKIP] reason=no_budget contributed=${c:.2f} "
                "realized=${r:.2f}"
            ).format(c=state.get("total_contributed_usd", 0.0), r=realized)
        )
        return signals

    usd = min(desired, budget_left)
    min_order = state.get("min_order_usd", 0.0)
    if usd < min_order:
        jp_debug(
            (
                "[JACKPOT][SKIP] reason=min_order desired=${d:.2f} budget=${b:.2f} "
                "min=${m:.2f}"
            ).format(d=desired, b=budget_left, m=min_order)
        )
        return signals

    if last_age_h < period_h:
        return signals

    qty = usd / price if price else 0.0
    signals.append(
        {
            "action": "buy",
            "usd": usd,
            "qty": qty,
            "price": price,
            "pos": pos,
            "multiplier": mult,
        }
    )
    avg_entry = state.get("avg_entry_price")
    contrib_total = state.get("total_contributed_usd", 0.0)
    inv_qty = state.get("inventory_qty", 0.0)
    next_in_h = max(period_h - last_age_h, 0.0)
    jp_trace(
        (
            "[JACKPOT][STATE] ATH=${ath:.4f} ATL=${atl:.4f} start=${start:.4f} "
            "take=${take:.4f} contrib=${c:.2f} inv_qty={q:.8f} avg=${avg:.4f} "
            "next_in={n:.1f}h"
        ).format(
            ath=ath,
            atl=atl,
            start=start_line,
            take=take_line,
            c=contrib_total,
            q=inv_qty,
            avg=avg_entry or 0,
            n=next_in_h,
        )
    )
    return signals


def apply_fills(state: dict, fills: List[dict], now_ts: int) -> None:
    """Apply executed fills to jackpot state."""
    for f in fills:
        action = f.get("action")
        if action == "buy":
            qty = f.get("qty", 0.0)
            usd = f.get("usd", 0.0)
            price = f.get("price", 0.0)
            inv = state.get("inventory_qty", 0.0)
            avg = state.get("avg_entry_price", 0.0)
            total_cost = inv * avg + qty * price
            new_qty = inv + qty
            state["inventory_qty"] = new_qty
            state["avg_entry_price"] = total_cost / new_qty if new_qty else 0.0
            state["total_contributed_usd"] += usd
            state["last_drip_ts"] = now_ts
            jp_info(f"[JACKPOT][BUY] qty={qty:.8f} usd=${usd:.2f} price=${price:.4f}")
            jp_info(
                (
                    "[JACKPOT][ADDED] +${usd:.2f} → contributed=${c:.2f} "
                    "inv_qty={q:.8f}"
                ).format(usd=usd, c=state["total_contributed_usd"], q=new_qty)
            )
            notify_telegram(
                (
                    "🚰 JACKPOT DRIP\n+${usd:.2f} at ${price:.4f}  (pos {pos:.2f}, mult {mult:.2f})\n"
                    "Contributed: ${c:.2f} | Inv: {q:.6f}"
                ).format(
                    usd=usd,
                    price=price,
                    pos=f.get("pos", 0.0),
                    mult=f.get("multiplier", 1.0),
                    c=state["total_contributed_usd"],
                    q=new_qty,
                )
            )
        elif action == "sell_all":
            qty = f.get("qty", 0.0)
            price = f.get("price", 0.0)
            proceeds = qty * price
            state["inventory_qty"] = 0.0
            state["avg_entry_price"] = 0.0
            jp_info(
                (
                    "[JACKPOT REACHED!!!] SELL_ALL qty={q:.8f} price=${p:.4f} "
                    "proceeds=${pr:.2f}"
                ).format(q=qty, p=price, pr=proceeds)
            )
            jp_info(
                (
                    "[JACKPOT REACHED!!!] TOTAL contributed=${c:.2f} "
                    "realized_pnl_cum=${r:.2f}"
                ).format(c=state.get("total_contributed_usd", 0.0), r=f.get("realized_cum", 0.0))
            )
            notify_telegram(
                (
                    "🎰 JACKPOT REACHED!!!\nSold {q:.6f} at ${p:.4f}  → Proceeds: ${pr:.2f}\n"
                    "Contributed total: ${c:.2f}\nRealized PnL (cum): ${r:.2f}"
                ).format(
                    q=qty,
                    p=price,
                    pr=proceeds,
                    c=state.get("total_contributed_usd", 0.0),
                    r=f.get("realized_cum", 0.0),
                )
            )
