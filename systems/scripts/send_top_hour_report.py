from __future__ import annotations

"""Generate and send a top-of-hour report from cached Kraken snapshots."""

import json
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from systems.utils.addlog import addlog, send_telegram_message
from systems.utils.path import find_project_root
from systems.utils.settings_loader import load_settings
from systems.utils.symbol_mapper import get_symbol_config


def _get_latest_price(trades: dict, pair: str) -> float:
    """Return the most recent trade price for ``pair`` from ``trades``."""
    if not isinstance(trades, dict):
        return 0.0
    latest = 0.0
    latest_time = -1.0
    for trade in trades.values():
        if trade.get("pair") != pair:
            continue
        ts = float(trade.get("time", 0.0))
        if ts > latest_time:
            latest_time = ts
            latest = float(trade.get("price", 0.0))
    return latest


def send_top_hour_report(
    ledger_name: str,
    tag: str,
    strategy_summary: dict,
    verbose: int = 0,
) -> None:
    """Load Kraken snapshot and send a formatted Telegram report."""
    root = find_project_root()
    snap_path = root / "data" / "snapshots" / f"{ledger_name}.json"
    if not snap_path.exists():
        addlog(
            f"[WARN] Snapshot for {ledger_name} not found; skipping report",
            verbose_int=1,
            verbose_state=verbose,
        )
        return

    try:
        with snap_path.open("r", encoding="utf-8") as f:
            snapshot = json.load(f)
    except Exception:
        addlog(
            f"[WARN] Failed to load snapshot for {ledger_name}",
            verbose_int=1,
            verbose_state=verbose,
        )
        return

    settings = load_settings()
    ledger_cfg = settings.get("ledger_settings", {}).get(ledger_name, {})
    symbol_cfg = get_symbol_config(tag)
    wallet_code = symbol_cfg["kraken"]["wallet_code"]
    fiat_code = ledger_cfg.get("fiat", symbol_cfg["kraken"]["fiat"])

    balance = snapshot.get("balance", {})
    trades = snapshot.get("trades", {})

    usd_balance = float(balance.get(fiat_code, 0.0))
    coin_balance = float(balance.get(wallet_code, 0.0))
    pair_code = symbol_cfg["kraken"]["wsname"].replace("/", "")
    price = _get_latest_price(trades, pair_code)
    coin_value = coin_balance * price
    total_value = usd_balance + coin_value

    # Determine display names
    coin_symbol = tag.replace("USD", "")
    fiat_symbol = fiat_code.replace("Z", "").replace("X", "")

    ct_now = datetime.now(ZoneInfo("America/Chicago")).strftime("%I:%M%p")
    lines = [f"🕒 {ct_now} CT | Ledger: {ledger_name}", ""]

    for name, data in strategy_summary.items():
        strat_name = name.title()
        if not strat_name.lower().endswith("catch"):
            strat_name += " Catch"
        buys = data.get("buys", 0)
        sells = data.get("sells", 0)
        open_n = data.get("open", 0)
        roi = data.get("roi", 0.0)
        total = data.get("total", 0.0)
        lines.append(
            " ".join(
                [
                    f"📈 {strat_name}",
                    f"| Buys: {buys}",
                    f"| Sells: {sells}",
                    f"| Open: {open_n}",
                    f"| ROI: {roi:+.1f}%",
                    f"| 💵 Total: ${total:+.2f}",
                ]
            )
        )

    lines.append("------------------------------------------------------------------------")
    lines.append(
        f"🧱 {fiat_symbol}: ${usd_balance:,.2f} | 🪙 {coin_symbol}: ${coin_value:,.2f} | "
        f"💰Total Wallet Value: ${total_value:,.2f}"
    )
    lines.append("------------------------------------------------------------------------")
    message = "\n".join(lines)
    addlog(message, verbose_int=1, verbose_state=verbose)
    send_telegram_message(message)
