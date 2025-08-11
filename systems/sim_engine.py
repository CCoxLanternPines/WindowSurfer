from __future__ import annotations

"""Simulation engine coordinating historical backtests.

This module wires together the configuration loader, data loader and
``TunnelManager`` to run a full historical simulation.  The engine is
stateless and always starts from a fresh ledger.
"""

from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Optional

import builtins
import pandas as pd

from .scripts.config_loader import load_runtime_config
from .scripts.data_loader import load_candle_history
from .scripts.tunnel_manager import TunnelManager


def _ledger_to_dict(ledger) -> Dict:
    """Serialize a :class:`~systems.scripts.ledger.Ledger` instance."""

    def _note_to_dict(note):
        d = asdict(note)
        d["timestamp"] = note.timestamp.isoformat()
        return d

    out: Dict[str, Dict[str, list]] = {}
    for sym, tunnel_map in ledger.notes.items():
        out[sym] = {}
        for tid, notes in tunnel_map.items():
            out[sym][tid] = [_note_to_dict(n) for n in notes]
    return out


def _save_ledger(ledger, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = _ledger_to_dict(ledger)
    import json

    with open(path, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2)


def run(
    ledger_name: str,
    *,
    start: Optional[str] = None,
    range_span: Optional[str] = None,
) -> Dict[str, float]:
    """Run a historical backtest for ``ledger_name``.

    Parameters
    ----------
    ledger_name:
        Name of the ledger configuration to load.
    start, range_span:
        Optional timespan expressions passed to
        :func:`load_candle_history` to window the data.

    Returns
    -------
    dict
        Summary statistics containing final capital, ROI and PnL.
    """

    cfg = load_runtime_config(ledger_name, runtime_mode="sim")
    capital = float(cfg.get("capital", 0.0))
    manager = TunnelManager(cfg, capital)

    coins = list(cfg.get("coins", {}).keys())
    history: Dict[str, pd.DataFrame] = {}
    for sym in coins:
        history[sym] = load_candle_history(sym, start, range_span)

    min_len = min((len(df) for df in history.values()), default=0)
    last_prices: Dict[str, float] = {sym: 0.0 for sym in coins}

    try:
        for idx in builtins.range(min_len):
            prices: Dict[str, float] = {}
            ts = None
            for sym, df in history.items():
                row = df.iloc[idx]
                prices[sym] = float(row["close"])
                ts = datetime.fromtimestamp(int(row["timestamp"]), tz=timezone.utc)
            last_prices = prices
            manager.tick(prices, ts or datetime.now(timezone.utc))
    except KeyboardInterrupt:
        pass

    ledger_path = Path("data") / "tmp" / f"{ledger_name}_sim.json"
    _save_ledger(manager.ledger, ledger_path)

    open_value = manager.ledger.total_fiat_value(last_prices)
    final_capital = manager.capital + open_value
    initial_capital = capital
    pnl = final_capital - initial_capital
    roi = (pnl / initial_capital) if initial_capital else 0.0

    summary = {
        "final_capital": final_capital,
        "roi": roi,
        "pnl": pnl,
    }

    print(
        f"Simulation completed. Final capital: {final_capital:.2f}, "
        f"ROI: {roi:.4f}, PnL: {pnl:.2f}"
    )

    return summary


__all__ = ["run"]
