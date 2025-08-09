from __future__ import annotations

"""Generate and send a top-of-hour report from cached Kraken snapshots."""

from datetime import datetime
from zoneinfo import ZoneInfo

from systems.utils.addlog import addlog, send_telegram_message
from systems.utils.config import load_settings
from systems.utils.resolve_symbol import split_tag
from systems.utils.snapshot import load_snapshot


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
    snapshot = load_snapshot(ledger_name)
    if not snapshot:
        addlog(
            f"[WARN] Snapshot for {ledger_name} not found; skipping report",
            verbose_int=1,
            verbose_state=verbose,
        )
        return

    settings = load_settings()
    ledger_cfg = settings.get("ledger_settings", {}).get(ledger_name, {})
    wallet_code = ledger_cfg.get("wallet_code", "")
    _, fiat_code = split_tag(tag)

    balance = snapshot.get("balance", {})
    trades = snapshot.get("trades", {})

    usd_balance = float(balance.get(fiat_code, 0.0))
    coin_balance = float(balance.get(wallet_code, 0.0))
    pair_code = tag
    price = _get_latest_price(trades, pair_code)
    coin_value = coin_balance * price
    total_value = usd_balance + coin_value

    # Determine display names
    fiat_symbol = fiat_code.replace("Z", "").replace("X", "")
    coin_symbol = (
        tag[: -len(fiat_symbol)] if fiat_symbol and tag.endswith(fiat_symbol) else tag
    )

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
