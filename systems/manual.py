from __future__ import annotations

"""Command-line tool to place manual Kraken buy/sell test orders."""

import argparse
import json
from typing import Optional

from systems.utils.settings_loader import load_settings
from systems.utils.path import find_project_root
from systems.utils.addlog import addlog
from systems.scripts.kraken_utils import ensure_snapshot, get_live_price
from systems.scripts.execution_handler import execute_buy, execute_sell


def _load_ledger(ledger_name: str) -> dict:
    root = find_project_root()
    path = root / "data" / "ledgers" / f"{ledger_name}.json"
    if path.exists():
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    return {"trades": []}


def _save_ledger(ledger_name: str, ledger: dict) -> None:
    root = find_project_root()
    path = root / "data" / "ledgers" / f"{ledger_name}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(ledger, f, indent=2)
def _coin_label(tag: str) -> str:
    for suffix in ["USD", "USDT", "USDC", "EUR", "GBP", "DAI"]:
        if tag.endswith(suffix):
            return tag[: -len(suffix)]
    return tag


def _parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Manual Kraken test trades")
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--buy", action="store_true", help="Execute a buy")
    action.add_argument("--sell", action="store_true", help="Execute a sell")
    parser.add_argument("--ledger", required=True, help="Ledger name from settings")
    parser.add_argument("--usd", required=True, type=float, help="USD amount")
    parser.add_argument("--dry", action="store_true", help="Simulate without execution")
    parser.add_argument("--verbose", type=int, default=1, help="Verbosity level")
    return parser.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> None:
    args = _parse_args(argv)

    settings = load_settings()
    ledger_cfg = settings.get("ledger_settings", {}).get(args.ledger)
    if ledger_cfg is None:
        raise SystemExit(f"[ERROR] Ledger '{args.ledger}' not found in settings")

    if args.usd <= 0:
        raise SystemExit("[ERROR] --usd must be positive")

    # Ensure snapshot exists
    snapshot = ensure_snapshot(args.ledger)
    if not snapshot:
        raise SystemExit(
            f"[ERROR] Snapshot unavailable for ledger '{args.ledger}'"
        )

    tag = ledger_cfg.get("tag")
    kraken_pair = ledger_cfg.get("kraken_name")
    fiat_code = ledger_cfg.get("fiat")

    price = get_live_price(kraken_pair)
    if price <= 0:
        raise SystemExit("[ERROR] Live price unavailable (0) — aborting")

    coin_amt = args.usd / price
    coin_str = _coin_label(tag)
    ledger = _load_ledger(args.ledger)

    if args.buy:
        if not args.dry:
            result = execute_buy(
                None,
                symbol=kraken_pair,
                fiat_code=fiat_code,
                price=price,
                amount_usd=args.usd,
                ledger_name=args.ledger,
                verbose=args.verbose,
            )
            if not result:
                raise SystemExit("[ERROR] Buy order failed")
            coin_amt = result.get("filled_amount", coin_amt)
            price = result.get("avg_price", price)
            ledger.setdefault("trades", []).append(
                {
                    "action": "buy",
                    "symbol": kraken_pair,
                    "usd": args.usd,
                    "coin": coin_amt,
                    "price": price,
                    "timestamp": result.get("timestamp"),
                }
            )
            _save_ledger(args.ledger, ledger)
        addlog(
            f"[MANUAL BUY] {args.ledger} | {tag} | ${args.usd:.2f} → {coin_amt:.4f} {coin_str} @ ${price:.4f}",
            verbose_int=1,
            verbose_state=args.verbose,
        )
    else:  # sell
        if not args.dry:
            result = execute_sell(
                None,
                symbol=kraken_pair,
                coin_amount=coin_amt,
                fiat_code=fiat_code,
                price=price,
                ledger_name=args.ledger,
                verbose=args.verbose,
            )
            if not result:
                raise SystemExit("[ERROR] Sell order failed")
            coin_amt = result.get("filled_amount", coin_amt)
            price = result.get("avg_price", price)
            usd_total = coin_amt * price
            ledger.setdefault("trades", []).append(
                {
                    "action": "sell",
                    "symbol": kraken_pair,
                    "usd": usd_total,
                    "coin": coin_amt,
                    "price": price,
                    "timestamp": result.get("timestamp"),
                }
            )
            _save_ledger(args.ledger, ledger)
        else:
            usd_total = args.usd
        addlog(
            f"[MANUAL SELL] {args.ledger} | {tag} | {coin_amt:.4f} {coin_str} → ${usd_total:.2f} @ ${price:.4f}",
            verbose_int=1,
            verbose_state=args.verbose,
        )


if __name__ == "__main__":
    main()
