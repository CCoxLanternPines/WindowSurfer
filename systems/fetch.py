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
from systems.utils.resolve_symbol import resolve_symbols, to_tag
from systems.scripts.fetch_candles import (
    fetch_binance_full_history_1h,
    fetch_kraken_last_n_hours_1h,
)


def run_fetch(
    account: str | None = None, market: str | None = None, all_accounts: bool = False
) -> None:
    """Fetch candles for configured accounts/markets."""

    cfg = load_config()
    accounts = cfg.get("accounts", {})
    targets = accounts.keys() if (all_accounts or not account) else [account]

    for acct_name in targets:
        acct_cfg = accounts.get(acct_name)
        if not acct_cfg:
            addlog(
                f"Error: Unknown account {acct_name}",
                verbose_int=1,
                verbose_state=True,
            )
            continue
        os.environ["WS_ACCOUNT"] = acct_name
        client = ccxt.kraken(
            {
                "enableRateLimit": True,
                "apiKey": acct_cfg.get("api_key", ""),
                "secret": acct_cfg.get("api_secret", ""),
            }
        )
        markets_cfg = acct_cfg.get("markets", {})
        m_targets = [market] if market else list(markets_cfg.keys())
        for m in m_targets:
            if m not in markets_cfg:
                continue
            addlog(
                f"[RUN][{acct_name}][{m}]",
                verbose_int=1,
                verbose_state=True,
            )
            symbols = resolve_symbols(client, m)
            kraken_name = symbols["kraken_name"]
            kraken_pair = symbols["kraken_pair"]
            binance_name = symbols["binance_name"]

            addlog(
                f"[RESOLVE][{acct_name}][{m}] KrakenName={kraken_name} KrakenPair={kraken_pair} BinanceName={binance_name}",
                verbose_int=1,
                verbose_state=True,
            )

            if "/" not in kraken_name:
                addlog(
                    f"[ERROR] Kraken symbol missing '/' : {kraken_name}",
                    verbose_int=1,
                    verbose_state=True,
                )
                raise SystemExit(1)
            if "/" in binance_name:
                addlog(
                    f"[ERROR] Binance symbol must not contain '/' : {binance_name}",
                    verbose_int=1,
                    verbose_state=True,
                )
                raise SystemExit(1)

            tag = to_tag(kraken_name)

            # Binance full history -> SIM
            df_sim = fetch_binance_full_history_1h(binance_name)
            sim_path = f"data/sim/{tag}_1h.csv"
            tmp_sim = sim_path + ".tmp"
            os.makedirs(os.path.dirname(sim_path), exist_ok=True)
            df_sim.to_csv(tmp_sim, index=False)
            os.replace(tmp_sim, sim_path)

            # Kraken last 720 -> LIVE
            df_live = fetch_kraken_last_n_hours_1h(kraken_name, n=720)
            live_path = f"data/live/{tag}_1h.csv"
            tmp_live = live_path + ".tmp"
            os.makedirs(os.path.dirname(live_path), exist_ok=True)
            df_live.to_csv(tmp_live, index=False)
            os.replace(tmp_live, live_path)
            rows = len(df_live)
            if rows < 720:
                addlog(
                    f"[FETCH][WARN] {acct_name} {kraken_name} returned {rows} rows (<720) from kraken",
                    verbose_int=1,
                    verbose_state=True,
                )

            addlog(
                f"[FETCH][{acct_name}][{kraken_name}] saved sim/live candles",
                verbose_int=1,
                verbose_state=True,
            )
