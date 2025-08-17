from __future__ import annotations

"""Historical simulation engine utilizing modular buy/sell evaluators."""

from typing import Dict, Any, List
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from systems.scripts.evaluate_buy import evaluate_buy
from systems.scripts.evaluate_sell import evaluate_sell
from systems.utils.config import load_settings
from systems.utils.resolve_symbol import (
    resolve_ccxt_symbols,
    to_tag,
    sim_path_csv,
)
from systems.utils.time import parse_cutoff


def run_simulation(*, ledger: str, verbose: int = 0, timeframe: str | None = None, viz: bool = True) -> None:
    """Run a basic backtest for ``ledger``.

    The function loads historical data, iterates over each candle and delegates
    buy/sell decisions to :func:`evaluate_buy` and :func:`evaluate_sell`.
    """

    settings = load_settings()
    ledger_cfg = settings.get("ledger_settings", {}).get(ledger)
    if not ledger_cfg:
        raise ValueError(f"Unknown ledger: {ledger}")

    kraken_symbol, _ = resolve_ccxt_symbols(settings, ledger)
    tag = to_tag(kraken_symbol)
    csv_path = sim_path_csv(tag)
    if not Path(csv_path).exists():
        raise FileNotFoundError(f"Missing data file: {csv_path}")

    df = pd.read_csv(csv_path)
    ts_col = next((c for c in df.columns if str(c).lower() in ("timestamp", "time", "date")), None)
    if ts_col is None:
        raise ValueError(f"No timestamp column in {csv_path}")
    df[ts_col] = pd.to_numeric(df[ts_col], errors="coerce")
    df = df.dropna(subset=[ts_col])
    if timeframe:
        try:
            delta = parse_cutoff(timeframe)
        except Exception:
            delta = None
        if delta:
            cutoff = (
                pd.Timestamp.utcnow().tz_localize(None) - delta
            ).timestamp()
            df = df[df[ts_col] >= cutoff]

    df = df.reset_index(drop=True)
    df["candle_index"] = range(len(df))

    state: Dict[str, Any] = {
        "buy_pressure": 0.0,
        "sell_pressure": 0.0,
        "open_notes": [],
        "realized_pnl": 0.0,
    }
    last_features: Dict[str, Any] | None = None
    closed_trades: List[Dict[str, Any]] = []
    buy_count = sell_count = flat_count = 0

    viz_ax = None
    if viz:
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots()
        ax.plot(df["candle_index"], df["close"], color="gray", zorder=1)
        viz_ax = ax

    for i in tqdm(range(len(df)), desc="📉 Sim Progress", dynamic_ncols=True):
        candle = df.iloc[i]

        before_open = len(state.get("open_notes", []))
        state, features = evaluate_buy(candle, last_features, state, viz_ax=viz_ax)
        if len(state.get("open_notes", [])) > before_open:
            buy_count += 1

        last_features = features
        slope_cls = features.get("slope_cls", 0) if features else 0

        before_sell = len(state.get("open_notes", []))
        state, closed = evaluate_sell(candle, slope_cls, state, viz_ax=viz_ax)
        if closed:
            if len(state.get("open_notes", [])) == 0 and before_sell > 0:
                sell_count += 1
            else:
                flat_count += 1
            closed_trades.extend(closed)

    if viz and viz_ax is not None:
        viz_ax.scatter([], [], color="green", marker="o", label="Buy")
        viz_ax.scatter([], [], color="red", marker="o", label="Sell")
        viz_ax.scatter([], [], color="orange", marker="o", label="Flat Sell")
        viz_ax.legend(loc="upper left")
        import matplotlib.pyplot as plt

        plt.show()

    print(f"[SIM] Realized PnL: {state['realized_pnl']:.2f}")
    print(f"[SIM] Open notes: {len(state['open_notes'])}")
    print(f"[SIM] Closed trades: {len(closed_trades)}")
    print(f"Buys={buy_count} Sells={sell_count} FlatSells={flat_count}")
