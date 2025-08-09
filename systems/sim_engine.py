"""Phase 0 simulator loop."""

from __future__ import annotations

from systems.scripts.get_candle_data import load_candles
from systems.scripts.ledger import Ledger


def run_sim(settings: dict) -> None:
    cfg = settings["phase0_settings"]
    capital = float(settings["simulation_capital"])
    tag = cfg["tag"]
    buy_interval = int(cfg["buy_interval"])
    buy_fraction = float(cfg["buy_fraction"])
    maturity_gain = float(cfg["maturity_gain"])

    candles = load_candles(tag)
    ledger = Ledger(maturity_gain)

    for step, candle in enumerate(candles):
        price = float(candle["close"])
        if step % buy_interval == 0:
            buy_usd = capital * buy_fraction
            if buy_usd > 0:
                amount = buy_usd / price
                capital -= buy_usd
                ledger.buy(price, amount, step)
        proceeds = ledger.check_sells(price, step)
        capital += proceeds

    last_price = float(candles[-1]["close"]) if candles else 0.0
    open_value = sum(n["amount"] * last_price for n in ledger.open_notes)
    final_capital = capital + open_value
    total_buys = len(ledger.closed_notes) + len(ledger.open_notes)
    total_sells = len(ledger.closed_notes)
    pnl = final_capital - float(settings["simulation_capital"])
    avg_hold = (
        sum(n["exit_step"] - n["entry_step"] for n in ledger.closed_notes) / total_sells
        if total_sells
        else 0
    )

    print(f"Final capital: {final_capital:.2f}")
    print(f"Total buys: {total_buys}")
    print(f"Total sells: {total_sells}")
    print(f"Total PnL: {pnl:.2f}")
    print(f"Average hold time: {avg_hold:.2f} candles")
