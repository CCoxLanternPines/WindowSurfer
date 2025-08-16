from __future__ import annotations

"""CLI helpers for candle fetching."""

import os
from pathlib import Path
import sys

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from systems.utils.addlog import addlog
from systems.utils.config import load_settings
from systems.utils.resolve_symbol import (
    resolve_ccxt_symbols,
    to_tag,
    live_path_csv,
    sim_path_csv,
)
from systems.scripts.fetch_candles import (
    fetch_binance_full_history_1h,
    fetch_kraken_last_n_hours_1h,
)


def run_fetch(ledger: str | None) -> None:
    """Fetch candles for ``ledger`` from Binance (SIM) and Kraken (LIVE)."""

    if not ledger:
        addlog(
            "Error: --ledger is required for fetch mode",
            verbose_int=1,
            verbose_state=True,
        )
        raise SystemExit(1)

    settings = load_settings()
    ledger_cfg = settings.get("ledger_settings", {}).get(ledger)
    if not ledger_cfg:
        addlog(
            f"Error: Unknown ledger {ledger}",
            verbose_int=1,
            verbose_state=True,
        )
        raise SystemExit(1)

    kraken_symbol, binance_symbol = resolve_ccxt_symbols(settings, ledger)

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

    tag = to_tag(kraken_symbol)

    # Binance full history -> SIM
    df_sim = fetch_binance_full_history_1h(binance_symbol)
    sim_path = sim_path_csv(tag)
    tmp_sim = sim_path + ".tmp"
    os.makedirs(os.path.dirname(sim_path), exist_ok=True)
    df_sim.to_csv(tmp_sim, index=False)
    os.replace(tmp_sim, sim_path)
    addlog(
        f"[FETCH][SIM] source=binance symbol={binance_symbol} tag={tag} rows={len(df_sim)} path={sim_path}",
        verbose_int=1,
        verbose_state=True,
    )

    # Kraken last 720 -> LIVE
    df_live = fetch_kraken_last_n_hours_1h(kraken_symbol, n=720)
    live_path = live_path_csv(tag)
    tmp_live = live_path + ".tmp"
    os.makedirs(os.path.dirname(live_path), exist_ok=True)
    df_live.to_csv(tmp_live, index=False)
    os.replace(tmp_live, live_path)
    rows = len(df_live)
    if rows < 720:
        addlog(
            f"[FETCH][LIVE][WARN] source=kraken symbol={kraken_symbol} returned {rows} rows (<720)",
            verbose_int=1,
            verbose_state=True,
        )
    addlog(
        f"[FETCH][LIVE] source=kraken symbol={kraken_symbol} tag={tag} rows={rows} path={live_path}",
        verbose_int=1,
        verbose_state=True,
    )
