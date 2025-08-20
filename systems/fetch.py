from __future__ import annotations

"""CLI helpers for candle fetching."""

import os
from pathlib import Path
import sys

import ccxt

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from systems.utils.addlog import addlog
from systems.utils.load_config import load_config
from systems.utils.resolve_symbol import resolve_symbols
from systems.scripts.fetch_candles import (
    fetch_binance_full_history_1h,
    fetch_kraken_last_n_hours_1h,
)


def run_fetch(account: str, market: str | None = None) -> None:
    """Fetch candles for markets defined under ``account``."""

    cfg = load_config()
    acct_cfg = cfg.get("accounts", {}).get(account)
    if not acct_cfg:
        addlog(
            f"Error: Unknown account {account}",
            verbose_int=1,
            verbose_state=True,
        )
        raise SystemExit(1)
    os.environ["WS_ACCOUNT"] = account
    markets = acct_cfg.get("markets", {})
    targets = [market] if market else list(markets.keys())

    client = ccxt.kraken(
        {
            "enableRateLimit": True,
            "apiKey": acct_cfg.get("api_key", ""),
            "secret": acct_cfg.get("api_secret", ""),
        }
    )

    for m in targets:
        symbols = resolve_symbols(client, m)
        kraken_symbol = symbols["kraken_name"]
        binance_symbol = symbols["binance_name"]

        if "/" not in kraken_symbol:
            addlog(
                f"[ERROR] Kraken symbol missing '/' : {kraken_symbol}",
                verbose_int=1,
                verbose_state=True,
            )
            raise SystemExit(1)
        if "/" in binance_symbol:
            addlog(
                f"[ERROR] Binance symbol must not contain '/' : {binance_symbol}",
                verbose_int=1,
                verbose_state=True,
            )
            raise SystemExit(1)

        file_tag = kraken_symbol.replace("/", "_")

        # Binance full history -> SIM
        df_sim = fetch_binance_full_history_1h(binance_symbol)
        sim_path = f"data/sim/{account}_{file_tag}_1h.csv"
        tmp_sim = sim_path + ".tmp"
        os.makedirs(os.path.dirname(sim_path), exist_ok=True)
        df_sim.to_csv(tmp_sim, index=False)
        os.replace(tmp_sim, sim_path)

        # Kraken last 720 -> LIVE
        df_live = fetch_kraken_last_n_hours_1h(kraken_symbol, n=720)
        live_path = f"data/live/{account}_{file_tag}_1h.csv"
        tmp_live = live_path + ".tmp"
        os.makedirs(os.path.dirname(live_path), exist_ok=True)
        df_live.to_csv(tmp_live, index=False)
        os.replace(tmp_live, live_path)
        rows = len(df_live)
        if rows < 720:
            addlog(
                f"[FETCH][WARN] {account} {kraken_symbol} returned {rows} rows (<720) from kraken",
                verbose_int=1,
                verbose_state=True,
            )

        addlog(
            f"[FETCH][{account}][{kraken_symbol}] saved sim/live candles",
            verbose_int=1,
            verbose_state=True,
        )
