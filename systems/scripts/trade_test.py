from __future__ import annotations

"""Utility to run a safe trade test for a configured account."""

from systems.utils.config import load_settings
from systems.scripts.fetch_candles import fetch_kraken_last_n_hours_1h
from systems.scripts.ledger import load_ledger, save_ledger
from systems.scripts.execution_handler import execute_buy, execute_sell
from systems.utils.resolve_symbol import resolve_symbols
import ccxt


def run_trade_test(account: str) -> None:
    """Execute a dry run of buy/sell operations for ``account``."""

    cfg = load_settings()
    acct_cfg = cfg.get("accounts", {}).get(account)
    if not acct_cfg:
        print(f"[ERROR] Unknown account {account}")
        return

    print(f"[TEST] Running trade test for {account}")

    client = ccxt.kraken({"enableRateLimit": True})
    for market, strat_cfg in acct_cfg.get("markets", {}).items():
        symbols = resolve_symbols(client, market)
        kraken_name = symbols["kraken_name"]
        kraken_pair = symbols["kraken_pair"]

        print(f"[TEST] Pulling 720h candles for {kraken_name}")
        df = fetch_kraken_last_n_hours_1h(kraken_name, n=720)
        if df.empty:
            print(f"[FAIL] No candles fetched for {kraken_name}")
            continue
        print(f"[PASS] Candles OK ({len(df)} rows)")

        ledger_name = f"{account}_{market.replace('/', '_')}"
        ledger_obj = load_ledger(ledger_name, tag=market.replace("/", "_"))
        save_ledger(ledger_name, ledger_obj, tag=market.replace("/", "_"))
        print(f"[PASS] Ledger load/save OK")

        price = float(df.iloc[-1]["close"])
        amt_usd = 10.0

        print(f"[TEST] Executing dummy BUY {amt_usd} at ~${price}")
        buy_res = execute_buy(
            None,
            pair_code=kraken_pair,
            price=price,
            amount_usd=amt_usd,
            ledger_name=ledger_name,
            wallet_code=acct_cfg.get("wallet_code"),
            verbose=1,
        )
        if not buy_res or buy_res.get("error"):
            print(f"[FAIL] Buy failed: {buy_res}")
            continue
        print(f"[PASS] Buy OK: {buy_res}")

        print(f"[TEST] Executing dummy SELL")
        sell_res = execute_sell(
            None,
            pair_code=kraken_pair,
            coin_amount=buy_res.get("filled_amount", 0.0),
            price=price,
            ledger_name=ledger_name,
            verbose=1,
        )
        if not sell_res or sell_res.get("error"):
            print(f"[FAIL] Sell failed: {sell_res}")
        else:
            print(f"[PASS] Sell OK: {sell_res}")

    print(f"[DONE] Trade test for {account}")

