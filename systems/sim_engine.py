from __future__ import annotations

from datetime import datetime, timezone, timedelta
import json
import os
import re

import ccxt
import pandas as pd
from tqdm import tqdm
from pathlib import Path

from systems.scripts.ledger import Ledger
from systems.scripts.candle_loader import load_candles_df
from systems.scripts.evaluate_buy import evaluate_buy
from systems.scripts.evaluate_sell import evaluate_sell
from systems.scripts.runtime_state import build_runtime_state
from systems.scripts.trade_apply import apply_buy, apply_sell
from systems.utils.addlog import addlog
from systems.utils.config import resolve_path, load_general, load_coin_settings, load_account_settings
from systems.utils.resolve_symbol import split_tag, resolve_symbols, to_tag
from systems.utils.trade_logger import init_logger as init_trade_logger

TIMEFRAME_SECONDS = {
    "s": 1,
    "m": 30 * 24 * 3600,
    "w": 7 * 24 * 3600,
    "d": 24 * 3600,
    "h": 3600,
}

def parse_timeframe(tf: str):
    if not tf:
        return None
    m = re.match(r"(?i)^\s*(\d+)\s*([smhdw])\s*$", tf)
    if not m:
        return None
    n, u = int(m.group(1)), m.group(2).lower()
    return pd.Timedelta(seconds=n * TIMEFRAME_SECONDS[u])

def _to_ccxt(market: str) -> str:
    market = market.upper()
    for quote in ("USDT", "USDC", "USD", "EUR", "GBP"):
        if market.endswith(quote):
            base = market[: -len(quote)]
            return f"{base}/{quote}"
    return market

def _run_single_sim(
    *,
    general,
    coin_settings,
    accounts_cfg,
    account: str,
    market: str,
    client: ccxt.Exchange,
    verbose: int = 0,
    timeframe: str = "1m",
    viz: bool = True,
) -> None:
    os.environ["WS_MODE"] = "sim"
    os.environ["WS_ACCOUNT"] = account

    market = market.replace("/", "").upper()
    ccxt_market = _to_ccxt(market)

    symbols = resolve_symbols(client, ccxt_market)
    kraken_name = symbols["kraken_name"]
    tag = to_tag(kraken_name)
    ledger_name = f"{account}_{market}"
    init_trade_logger(ledger_name)
    base, _ = split_tag(tag)

    if account.upper() == "SIM":
        csv_path = Path("data/candles/sim") / f"{market}.csv"
        if not csv_path.exists():
            raise FileNotFoundError(f"Missing data file: {csv_path}")
        df = pd.read_csv(csv_path)
        ts_col = "timestamp" if "timestamp" in df.columns else df.columns[0]
    else:
        csv_path = Path("data/candles/sim") / f"{market}.csv"
        df, _ = load_candles_df(account, market, verbose=verbose)
        ts_col = "timestamp"

    last_ts_raw = float(df[ts_col].iloc[-1]) if len(df) else 0.0
    is_ms = last_ts_raw > 1e12
    TFACTOR = 1000.0 if is_ms else 1.0

    if timeframe:
        delta = parse_timeframe(timeframe)
        if delta:
            end_ts = float(df[ts_col].iloc[-1])
            start_ts = end_ts - delta.total_seconds() * TFACTOR
            df = df[df[ts_col] >= start_ts].reset_index(drop=True)

    ledger_obj = Ledger()
    start_capital = float(general.get("simulation_capital", 0.0))
    ledger_obj.set_metadata({"capital": start_capital})

    runtime_state = build_runtime_state(
        general, coin_settings, accounts_cfg, account, ccxt_market, "sim", client=client
    )

    for t in tqdm(range(len(df)), desc="📉 Sim Progress", unit="tick"):
        candle = df.iloc[t]
        buy_signal = evaluate_buy(ctx={}, t=t, series=df, cfg=coin_settings.get("default", {}), runtime_state=runtime_state)
        if buy_signal:
            units = buy_signal.get("units") or (buy_signal["size_usd"] / buy_signal["entry_price"])
            apply_buy(
                ledger=ledger_obj,
                window_name="default",
                t=t,
                meta=buy_signal,
                result={
                    "filled_amount": units,
                    "avg_price": buy_signal["entry_price"],
                    "timestamp": int(candle["timestamp"]),
                },
                state=runtime_state,
            )
            runtime_state["capital"] -= buy_signal["size_usd"]

        open_notes = ledger_obj.get_open_notes()
        sell_candidates = evaluate_sell(
            ctx={}, t=t, series=df, cfg=coin_settings.get("default", {}), open_notes=open_notes, runtime_state=runtime_state
        )
        for sig in sell_candidates:
            result = {
                "filled_amount": sig["units"],
                "avg_price": sig["sell_price"],
                "timestamp": sig["created_ts"],
            }
            apply_sell(
                ledger=ledger_obj,
                note=sig,
                t=t,
                result=result,
                state=runtime_state,
            )

    temp_dir = resolve_path("data/temp")
    temp_dir.mkdir(parents=True, exist_ok=True)
    full_path = temp_dir / "sim_data.json"

    entries = []
    for note in ledger_obj.get_closed_notes():
        entries.append({
            "idx": note.get("created_idx"),
            "price": note.get("exit_price", note.get("entry_price")),
            "side": "SELL",
            "usd": note.get("exit_price", 0.0) * note.get("entry_amount", 0.0),
        })
    for note in ledger_obj.get_open_notes():
        entries.append({
            "idx": note.get("created_idx"),
            "price": note.get("entry_price"),
            "side": "BUY",
            "usd": note.get("entry_price", 0.0) * note.get("entry_amount", 0.0),
        })

    summary = ledger_obj.get_account_summary(float(df["close"].iloc[-1]))
    simple_ledger = {
        "trades": entries,
        "meta": {
            "coin": market,
            "start_capital": start_capital,
            "final_value": summary.get("total_value", 0.0),
        },
        "series": {
            "timestamps": df["timestamp"].tolist(),
            "close": df["close"].tolist(),
        },
    }
    with full_path.open("w", encoding="utf-8") as fh:
        json.dump(simple_ledger, fh, indent=2)

def run_simulation(*, coin: str, timeframe: str = "1m", viz: bool = True) -> None:
    general = load_general()
    coin_settings = load_coin_settings()
    accounts_cfg = load_account_settings()

    market = coin.replace("/", "").upper()
    account = "SIM"
    client = ccxt.kraken({"enableRateLimit": True})

    _run_single_sim(
        general=general,
        coin_settings=coin_settings,
        accounts_cfg=accounts_cfg,
        account=account,
        market=market,
        client=client,
        verbose=0,
        timeframe=timeframe,
        viz=viz,
    )
