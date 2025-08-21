from __future__ import annotations
"""Tiny live smoke trade: buy then sell to verify keys."""

import time
from typing import Any

import ccxt

from systems.utils.config import (
    load_general,
    load_account_settings,
    load_coin_settings,
    load_keys,
)
from systems.utils.quote_norm import assert_quote_match
from systems.scripts.fetch_candles import fetch_candles  # noqa: F401
from systems.utils.addlog import addlog


def _fail(msg: str) -> int:
    print(f"[SMOKE][FAIL] {msg}")
    return 1


def run_smoke_trade(
    account: str,
    market: str,
    budget: float,
    confirm: str,
    verbose: int = 0,
) -> int:
    # 0) Preconditions
    if confirm != "LIVE":
        return _fail("Missing --confirm LIVE (guard)")
    accounts = load_account_settings()
    acct = accounts.get(account)
    if not acct:
        return _fail(f"Unknown account '{account}'")
    if not acct.get("is_live", False):
        return _fail(f"Account '{account}' is not live (set is_live=true)")
    if market not in (acct.get("market settings") or {}):
        return _fail(f"Market '{market}' not enabled for account '{account}'")
    keys = load_keys()
    kp = keys.get(account)
    if not kp or not kp.get("api_key") or not kp.get("api_secret"):
        return _fail("Missing API keys for account")

    # 1) Exchange client
    client = ccxt.kraken(
        {
            "enableRateLimit": True,
            "apiKey": kp["api_key"],
            "secret": kp["api_secret"],
        }
    )

    # 2) Validate pair + quote consistency and min requirements
    try:
        market_info = client.load_markets()[market]
    except Exception:
        return _fail(f"Exchange does not list market '{market}'")

    quote = market_info["quote"]
    base = market_info["base"]
    assert_quote_match(quote_expected=quote, exchange_pair=market)

    min_cost = float(market_info.get("limits", {}).get("cost", {}).get("min") or 0)
    min_amount = float(market_info.get("limits", {}).get("amount", {}).get("min") or 0)

    if budget <= 0:
        return _fail("Budget must be > 0")
    if min_cost and budget < min_cost:
        return _fail(f"Budget ${budget:.2f} < exchange min notional ${min_cost:.2f}")

    # 3) Get last price, compute size
    ticker = client.fetch_ticker(market)
    last = float(ticker["last"])
    if last <= 0:
        return _fail("Invalid last price from exchange")
    size = budget / last
    precision = market_info.get("precision", {}).get("amount", 8)
    size = round(size, precision)
    if min_amount and size < min_amount:
        size = min_amount
        eff_cost = size * last
        if eff_cost > budget * 1.25:
            return _fail(f"Budget too small for min size; need ≈ ${eff_cost:.2f}")

    if verbose:
        print(f"[SMOKE] {market} last={last} size={size} est_cost≈{size*last:.2f}")

    # 4) Place market BUY
    try:
        buy = client.create_market_buy_order(market, size)
    except Exception as e:
        return _fail(f"BUY failed: {e}")
    buy_id = buy.get("id") or buy.get("order", {}).get("id")
    print(f"[SMOKE][BUY][OK] id={buy_id} size={size}")

    # 5) Wait a beat, then SELL the filled amount
    time.sleep(2)
    filled = float(buy.get("filled") or buy.get("amount") or size)
    try:
        sell = client.create_market_sell_order(market, filled)
    except Exception as e:
        return _fail(f"SELL failed: {e}")
    sell_id = sell.get("id") or sell.get("order", {}).get("id")
    print(f"[SMOKE][SELL][OK] id={sell_id} size={filled}")

    # 6) Summarize fees / PnL
    buy_cost = float(buy.get("cost") or 0.0)
    sell_cost = float(sell.get("cost") or 0.0)
    pnl_quote = sell_cost - buy_cost if (buy_cost and sell_cost) else 0.0
    print(
        f"[SMOKE][DONE] base={base} quote={quote} cost≈{buy_cost:.2f} -> {sell_cost:.2f} Δ≈{pnl_quote:.2f}"
    )

    return 0

