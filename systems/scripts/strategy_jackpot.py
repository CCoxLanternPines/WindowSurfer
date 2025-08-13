from __future__ import annotations

"""Jackpot drip and cashout strategy helper."""

from typing import Any, Dict

from systems.utils.addlog import addlog, send_telegram_message
from systems.scripts.trade_apply import (
    paper_execute_buy,
    paper_execute_sell,
    apply_buy_result_to_ledger,
    apply_sell_result_to_ledger,
)
from systems.scripts.execution_handler import execute_buy, execute_sell


def _parse_period(period: str) -> int:
    """Return number of seconds for ``period`` like "1w"."""
    if not period:
        return 0
    try:
        value = int(period[:-1])
    except ValueError:
        return 0
    unit = period[-1].lower()
    if unit == "h":
        factor = 3600
    elif unit == "d":
        factor = 3600 * 24
    elif unit == "w":
        factor = 3600 * 24 * 7
    else:
        factor = 0
    return value * factor


def _compute_global_p(jstate: Dict[str, Any], price: float) -> float:
    low = float(jstate.get("global_low", 0.0))
    high = float(jstate.get("global_high", 0.0))
    if price > high:
        addlog(
            f"[JACKPOT][STALE_BOUNDS] current_high={price:.2f} global_high={high:.2f}"
        )
        jstate["global_high"] = price
        high = price
    elif price < low:
        addlog(
            f"[JACKPOT][STALE_BOUNDS] current_low={price:.2f} global_low={low:.2f}"
        )
        jstate["global_low"] = price
        low = price
    if high == low:
        return 0.0
    p = (price - low) / (high - low)
    return max(0.0, min(1.0, p))


def init_jackpot(state: Dict[str, Any], ledger_cfg: Dict[str, Any], df) -> None:
    cfg = ledger_cfg.get("jackpot", {})
    enabled = bool(cfg.get("enabled"))
    period_s = _parse_period(cfg.get("period", "1w"))
    jstate = state.get("jackpot", {})
    jstate.setdefault("pool_usd", 0.0)
    jstate.setdefault("notes_open", [])
    jstate.setdefault("drips", 0.0)
    jstate.setdefault("buys", 0)
    jstate.setdefault("sells", 0)
    jstate.setdefault("realized_pnl", 0.0)
    jstate["enabled"] = enabled
    jstate["cfg"] = cfg
    jstate["period_s"] = period_s
    if df is not None and len(df):
        if "low" in df:
            jstate["global_low"] = float(df["low"].min())
        else:
            jstate["global_low"] = float(df["close"].min())
        if "high" in df:
            jstate["global_high"] = float(df["high"].max())
        else:
            jstate["global_high"] = float(df["close"].max())
        if "timestamp" in df.columns:
            now_ts = int(df.iloc[-1]["timestamp"])
        else:
            now_ts = 0
        jstate.setdefault("next_period_ts", ((now_ts // period_s) + 1) * period_s if period_s else 0)
    state["jackpot"] = jstate


def on_buy_drip(state: Dict[str, Any], buy_usd: float) -> float:
    j = state.get("jackpot", {})
    if not j.get("enabled"):
        return buy_usd
    frac = float(j.get("cfg", {}).get("drip_fraction", 0.0))
    drip = buy_usd * frac
    if drip <= 0:
        return buy_usd
    j["pool_usd"] = j.get("pool_usd", 0.0) + drip
    j["drips"] = j.get("drips", 0.0) + drip
    if state.get("ctx", {}).get("verbosity", 0) >= 2:
        addlog(f"[JACKPOT][DRIP] +${drip:.2f} → pool=${j['pool_usd']:.2f}")
    return buy_usd - drip


def maybe_periodic_jackpot_buy(ctx: Dict[str, Any], state: Dict[str, Any], t: int, df, price: float, limits: Dict[str, float], ledger_tag: str) -> None:
    j = state.get("jackpot", {})
    if not j.get("enabled"):
        return
    if "timestamp" in df.columns:
        now_ts = int(df.iloc[t]["timestamp"])
    else:
        now_ts = 0
    if now_ts < j.get("next_period_ts", 0):
        return
    p_global = _compute_global_p(j, price)
    cfg = j.get("cfg", {})
    trigger_p = float(cfg.get("trigger_p", 0.5))
    min_usd = float(cfg.get("min_usd", 0.0))
    if p_global <= trigger_p and j.get("pool_usd", 0.0) >= min_usd:
        depth_scale = float(cfg.get("depth_scale", 1.0))
        mult = 1.0 + (trigger_p - p_global) * depth_scale
        amount_usd_calc = j["pool_usd"] * mult
        max_note = float(limits.get("max_note_usdt", amount_usd_calc))
        pool = float(j.get("pool_usd", 0.0))
        amount_usd = max(0.0, min(pool, amount_usd_calc, max_note))
        min_note = float(limits.get("min_note_size", 0.0))
        if amount_usd <= 0 or amount_usd < min_note:
            if ctx.get("verbosity", 0) >= 3:
                addlog(
                    f"[JACKPOT][POOL_DEBIT][SKIP] pool=${pool:.2f} "
                    f"calc=${amount_usd_calc:.2f} cap=${max_note:.2f}"
                )
            j["next_period_ts"] = j.get("next_period_ts", 0) + j.get("period_s", 0)
            return
        j["pool_usd"] = pool - amount_usd
        if ctx.get("verbosity", 0) >= 1:
            addlog(f"[JACKPOT][POOL_DEBIT] -${amount_usd:.2f} → pool=${j['pool_usd']:.2f}")
        mode = state.get("mode", "sim")
        ts = now_ts if now_ts else None
        note = None
        if mode == "live":
            result = execute_buy(
                None,
                pair_code=ctx.get("pair_code", ledger_tag),
                price=price,
                amount_usd=amount_usd,
                ledger_name=ledger_tag,
                wallet_code=ctx.get("wallet_code", ""),
                verbose=state.get("verbose", 0),
            )
            if result and not result.get("error"):
                note = apply_buy_result_to_ledger(
                    ledger=ctx["ledger"],
                    window_name="jackpot",
                    t=t,
                    meta={"kind": "jackpot", "window_name": "jackpot", "window_size": cfg.get("period", "")},
                    result=result,
                    state=state,
                )
        else:
            result = paper_execute_buy(price, amount_usd, timestamp=ts)
            note = apply_buy_result_to_ledger(
                ledger=ctx["ledger"],
                window_name="jackpot",
                t=t,
                meta={"kind": "jackpot", "window_name": "jackpot", "window_size": cfg.get("period", "")},
                result=result,
                state=state,
            )
        if note:
            j.setdefault("notes_open", []).append(note)
            j["buys"] = j.get("buys", 0) + 1
            msg = f"[JACKPOT][BUY] p={p_global:.3f} pool→${j['pool_usd']:.2f} spent ${amount_usd:.2f}"
            addlog(msg)
            send_telegram_message(f"🎰 {msg}")
    j["next_period_ts"] = j.get("next_period_ts", 0) + j.get("period_s", 0)


def maybe_cashout_jackpot(ctx: Dict[str, Any], state: Dict[str, Any], t: int, df, price: float, limits: Dict[str, float], ledger_tag: str) -> None:
    j = state.get("jackpot", {})
    if not j.get("enabled"):
        return
    cfg = j.get("cfg", {})
    cashout_p = float(cfg.get("cashout_p", 1.0))
    p_global = _compute_global_p(j, price)
    if ctx.get("verbosity", 0) >= 2:
        addlog(
            f"[JACKPOT][P_DEBUG] now=${price:.2f}, low=${j.get('global_low', 0.0):.2f}, "
            f"high=${j.get('global_high', 0.0):.2f}, p={p_global:.3f}, cashout_p={cashout_p:.3f}"
        )
    notes = list(j.get("notes_open", []))
    if not notes:
        if ctx.get("verbosity", 0) >= 3:
            addlog("[JACKPOT][NONE_TO_SELL]")
        return
    
    if p_global < cashout_p:
        if ctx.get("verbosity", 0) >= 2:
            addlog(f"[JACKPOT][NO_TRIGGER] p={p_global:.3f} < cashout_p={cashout_p:.3f}")
        return
    ts = int(df.iloc[t]["timestamp"]) if "timestamp" in df.columns else None
    sold = 0
    total_gain = 0.0
    for note in notes:
        # Ensure we only process jackpot notes
        if note.get("kind") != "jackpot":
            addlog(
                f"[JACKPOT][DEBUG] skip note id={note.get('id')} kind={note.get('kind')}"
            )
            continue

        # Skip notes that are too small to sell
        value = note.get("entry_amount", 0.0) * price
        min_size = float(limits.get("min_note_size", 0.0))
        if value < min_size:
            addlog(
                f"[JACKPOT][SKIP_MIN] id={note.get('id')} value=${value:.2f} min=${min_size:.2f}"
            )
            continue

        mode = state.get("mode", "sim")
        if mode == "live":
            result = execute_sell(
                None,
                pair_code=ctx.get("pair_code", ledger_tag),
                coin_amount=note.get("entry_amount", 0.0),
                price=price,
                ledger_name=ledger_tag,
                verbose=state.get("verbose", 0),
            )
            if not result or result.get("error"):
                continue
        else:
            result = paper_execute_sell(price, note.get("entry_amount", 0.0), timestamp=ts)
        apply_sell_result_to_ledger(
            ledger=ctx["ledger"],
            note=note,
            t=t,
            result=result,
            state=state,
        )
        exit_usd = note.get(
            "exit_usdt", result.get("filled_amount", 0.0) * result.get("avg_price", 0.0)
        )
        proceeds = float(result.get("proceeds_usd", exit_usd))
        j["pool_usd"] = float(j.get("pool_usd", 0.0)) + proceeds
        if ctx.get("verbosity", 0) >= 3:
            addlog(
                f"[JACKPOT][POOL_CREDIT] +${proceeds:.2f} → pool=${j['pool_usd']:.2f}"
            )
        qty = result.get("filled_amount", note.get("entry_amount", 0.0))
        exit_price = note.get("exit_price", result.get("avg_price", 0.0))
        addlog(
            f"[JACKPOT][SELL] id={note.get('id')} qty={qty:.8f} "
            f"price={exit_price:.2f} pnl=${note.get('gain', 0.0):.2f}]"
        )
        total_gain += note.get("gain", 0.0)
        sold += 1
        j["notes_open"].remove(note)
    if sold:
        j["sells"] = j.get("sells", 0) + sold
        j["realized_pnl"] = j.get("realized_pnl", 0.0) + total_gain
        msg = f"[JACKPOT][CASHOUT] sold {sold} notes @ p={p_global:.3f} gain=${total_gain:.2f}"
        addlog(msg)
        send_telegram_message(f"🎰 {msg}")
    if ctx.get("verbosity", 0) >= 3:
        jackpot_notes = [n for n in j.get("notes_open", []) if n.get("kind") == "jackpot"]
        coin_value = sum(
            (float(n.get("entry_amount", 0.0)) or 0.0) * float(price)
            for n in jackpot_notes
        )
        pool_now = float(j.get("pool_usd", 0.0))
        addlog(
            f"[JACKPOT][AUDIT] pool_usd=${pool_now:.2f} coin_value=${coin_value:.2f} "
            f"total=${(pool_now + coin_value):.2f} open_notes={len(jackpot_notes)}"
        )
