#!/usr/bin/env python3
"""Connectivity and configuration checks.

Example:
    python test.py --account Kris --market DOGEUSD --smoketest

Use ``--help`` to see available options.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

import ccxt  # type: ignore


def _normalize_market(market: str) -> str:
    if "/" in market:
        return market
    market = market.upper()
    for quote in ("USDT", "USDC", "USD", "EUR", "GBP"):
        if market.endswith(quote):
            base = market[: -len(quote)]
            return f"{base}/{quote}"
    return market


def _load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def main(argv: Optional[list[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Configuration tester")
    parser.add_argument("--account", required=True, help="Account name")
    parser.add_argument("--market", required=True, help="Market symbol e.g. DOGEUSD")
    parser.add_argument("--smoketest", action="store_true", help="Run exchange ping")
    args = parser.parse_args(argv)

    ok = True

    acct_cfg = _load_json(Path("settings/account_settings.json")).get("accounts", {})
    coin_cfg = _load_json(Path("settings/coin_settings.json")).get("coin_settings", {})

    account_ok = args.account in acct_cfg
    market_ok = account_ok and args.market in acct_cfg.get(args.account, {}).get("market settings", {})
    coin_ok = args.market in coin_cfg or "default" in coin_cfg

    keys_path = Path("settings/keys.json")
    secrets_ok = False
    if keys_path.exists():
        keys = _load_json(keys_path)
        acct_keys = keys.get(args.account, {})
        secrets_ok = bool(acct_keys.get("api_key")) and bool(acct_keys.get("api_secret"))
    else:
        acct = acct_cfg.get(args.account, {})
        kraken_api = acct.get("kraken_api", {})
        secrets_ok = bool(kraken_api.get("key")) and bool(kraken_api.get("secret"))

    if not account_ok:
        print(f"❌ Unknown account {args.account}")
        ok = False
    if not market_ok:
        print(f"❌ Unknown market {args.market} for account {args.account}")
        ok = False
    if not coin_ok:
        print(f"❌ Missing coin settings for {args.market}")
        ok = False
    if not secrets_ok:
        print(f"❌ Missing API secrets for account {args.account}")
        ok = False

    if args.smoketest and ok:
        exchange = ccxt.kraken({"enableRateLimit": True})
        try:
            exchange.public_get_time()
            exchange.fetch_ticker(_normalize_market(args.market))
            print("✅ Kraken connectivity ok")
        except Exception as exc:
            print(f"❌ Kraken connectivity failed: {exc}")
            ok = False

    sys.exit(0 if ok else 1)


if __name__ == "__main__":  # pragma: no cover
    main()
