from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict


def run_sim(config: Dict[str, int | float | dict]) -> Dict[str, float]:
    """Run a minimal trading simulation and return metrics.

    The simulation loads historical candle data from ``data/raw/DOGEUSD.csv``
    and executes a simple strategy based on the provided configuration.
    """

    capital = float(config["simulation_capital"])
    buy_cfg = config["buy_settings"]
    sell_cfg = config["sell_settings"]

    buy_frequency = int(buy_cfg["buy_frequency"])
    buy_fraction = float(buy_cfg["buy_fraction"])
    max_notes = int(buy_cfg["max_concurrent_notes"])
    dip_depth_entry = float(buy_cfg["dip_depth_entry"])
    slope_filter = float(buy_cfg["slope_filter"])
    volatility_filter = float(buy_cfg["volatility_filter"])

    min_hold_time = int(sell_cfg["min_hold_time"])
    maturity_gain_target = float(sell_cfg["maturity_gain_target"])
    max_hold_time = int(sell_cfg["max_hold_time"])
    stop_loss_pct = float(sell_cfg["stop_loss_pct"])
    trailing_stop_pct = float(sell_cfg["trailing_stop_pct"])
    trailing_start_gain = float(sell_cfg["trailing_start_gain"])

    data_path = Path("data/raw/DOGEUSD.csv")
    prices: list[float] = []
    with data_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            prices.append(float(row["close"]))

    open_notes: list[dict] = []
    prev_price = prices[0] if prices else 0.0

    for step, price in enumerate(prices):
        slope = (price - prev_price) / prev_price if prev_price else 0.0
        volatility = abs(price - prev_price) / prev_price if prev_price else 0.0

        # update peak price for trailing stop
        for note in open_notes:
            if price > note["peak_price"]:
                note["peak_price"] = price

        # check sells
        notes_to_close: list[dict] = []
        for note in open_notes:
            hold_time = step - note["entry_step"]
            gain = (price - note["entry_price"]) / note["entry_price"]
            peak_gain = (note["peak_price"] - note["entry_price"]) / note["entry_price"]
            drop_from_peak = (note["peak_price"] - price) / note["peak_price"] if note["peak_price"] else 0.0

            if (hold_time >= min_hold_time and gain >= maturity_gain_target) or hold_time >= max_hold_time:
                notes_to_close.append(note)
            elif gain <= -stop_loss_pct:
                notes_to_close.append(note)
            elif peak_gain >= trailing_start_gain and drop_from_peak >= trailing_stop_pct:
                notes_to_close.append(note)

        for note in notes_to_close:
            capital += note["amount"] * price
            open_notes.remove(note)

        # check buys
        if (
            step % buy_frequency == 0
            and len(open_notes) < max_notes
            and capital > 0
            and prev_price > 0
            and (prev_price - price) / prev_price >= dip_depth_entry
            and slope <= slope_filter
            and volatility <= volatility_filter
        ):
            usd_to_spend = capital * buy_fraction
            if usd_to_spend > 0:
                amount = usd_to_spend / price
                capital -= usd_to_spend
                open_notes.append(
                    {
                        "entry_price": price,
                        "amount": amount,
                        "entry_step": step,
                        "peak_price": price,
                    }
                )

        prev_price = price

    final_price = prices[-1] if prices else 0.0
    for note in open_notes:
        capital += note["amount"] * final_price

    final_capital = capital
    pnl = final_capital - float(config["simulation_capital"])
    return {"final_capital": final_capital, "pnl": pnl}
