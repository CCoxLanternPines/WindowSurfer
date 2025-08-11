from __future__ import annotations

"""Live trading engine coordinating hourly decision making.

The live engine waits for the top of each UTC hour, fetches the latest
candle close prices from Kraken, passes them to the ``TunnelManager`` and
optionally executes resulting orders via ``execution_handler``.  The
ledger state is persisted after every tick so the engine can be safely
restarted.
"""

import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict

import requests

from .scripts.config_loader import load_runtime_config
from .scripts.tunnel_manager import TunnelManager
from .scripts import execution_handler as eh


_API_BASE = "https://api.kraken.com/0/public/OHLC"


def _save_ledger(ledger, ledger_name: str) -> None:
    from dataclasses import asdict
    import json

    path = Path("data") / "tmp" / f"{ledger_name}_live.json"
    path.parent.mkdir(parents=True, exist_ok=True)

    def _note_to_dict(note):
        d = asdict(note)
        d["timestamp"] = note.timestamp.isoformat()
        return d

    data = {}
    for sym, tunnels in ledger.notes.items():
        data[sym] = {tid: [_note_to_dict(n) for n in notes] for tid, notes in tunnels.items()}

    with open(path, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2)


def _wait_to_top_of_hour() -> None:
    now = datetime.utcnow()
    seconds = 3600 - (now.minute * 60 + now.second)
    if seconds > 0:
        time.sleep(seconds)


def _fetch_close(pair: str, retries: int = 3) -> float:
    for _ in range(retries):
        try:
            resp = requests.get(_API_BASE, params={"pair": pair, "interval": 60}, timeout=10)
            resp.raise_for_status()
            data = resp.json()["result"]
            # Kraken sometimes returns different pair keys; grab the first
            k = next(iter(data))
            close = float(data[k][-1][4])
            return close
        except Exception:
            time.sleep(1)
    raise RuntimeError(f"failed to fetch price for {pair}")


def run(ledger_name: str, *, dry_run: bool = False) -> None:
    cfg = load_runtime_config(ledger_name, runtime_mode="live")
    capital = float(cfg.get("capital", 0.0))
    manager = TunnelManager(cfg, capital)

    exec_handler = eh

    coins = cfg.get("coins", {})

    while True:
        _wait_to_top_of_hour()
        prices: Dict[str, float] = {}
        for sym, coin_cfg in coins.items():
            pair = coin_cfg["kraken"]["pair"]
            prices[sym] = _fetch_close(pair)
        timestamp = datetime.now(timezone.utc)
        actions = manager.tick(prices, timestamp)

        # Execute resulting orders
        for sym, logs in actions.items():
            pair_cfg = coins[sym]["kraken"]
            for log in logs:
                parts = log.split()
                if len(parts) < 5:
                    continue
                side, _, qty, _, _price = parts
                qty_f = float(qty)
                if dry_run:
                    continue
                try:
                    if side == "buy":
                        exec_handler.place_market_buy(
                            pair_cfg["pair"],
                            qty_f,
                            precision=pair_cfg.get("quantity_precision"),
                            min_qty=pair_cfg.get("min_order_coin"),
                        )
                    elif side == "sell":
                        exec_handler.place_market_sell(
                            pair_cfg["pair"],
                            qty_f,
                            precision=pair_cfg.get("quantity_precision"),
                            min_qty=pair_cfg.get("min_order_coin"),
                        )
                except Exception:
                    # In a live environment we'd log this; for now we swallow
                    pass
        _save_ledger(manager.ledger, ledger_name)


__all__ = ["run"]
