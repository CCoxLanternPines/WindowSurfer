from __future__ import annotations

"""Command-line tool to place manual Kraken buy/sell test orders."""

import json
from typing import Optional

from systems.utils.config import load_settings, resolve_path
from systems.utils.addlog import addlog
from systems.scripts.kraken_utils import ensure_snapshot, get_live_price
from systems.utils.cli import build_parser
from systems.scripts.execution_handler import execute_buy, execute_sell
from systems.scripts.ledger import save_ledger


def _load_ledger(ledger_name: str) -> dict:
    path = resolve_path(f"data/ledgers/{ledger_name}.json")
    if path.exists():
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    return {"trades": []}
def _coin_label(tag: str) -> str:
    for suffix in ["USD", "USDT", "USDC", "EUR", "GBP", "DAI"]:
        if tag.endswith(suffix):
            return tag[: -len(suffix)]
    return tag


def _parse_args(argv: Optional[list[str]] = None):
    parser = build_parser()
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--buy", action="store_true", help="Execute a buy")
    action.add_argument("--sell", action="store_true", help="Execute a sell")
    parser.add_argument("--usd", required=True, type=float, help="USD amount")
    args = parser.parse_args(argv)
    if not args.ledger:
        parser.error("--ledger is required")
    return args


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

    price = get_live_price(tag)
    if price <= 0:
        raise SystemExit("[ERROR] Live price unavailable (0) — aborting")

    coin_amt = args.usd / price
    coin_str = _coin_label(tag)
    ledger = _load_ledger(args.ledger)

    if args.buy:
        if not args.dry:
            result = execute_buy(
                None,
                pair_code=ledger_cfg["kraken_pair"],
                wallet_code=ledger_cfg["wallet_code"],
                price=price,
                amount_usd=args.usd,
                ledger_name=args.ledger,
                verbose=args.verbose,
            )
            if not result or result.get("error"):
                raise SystemExit("[ERROR] Buy order failed")
            coin_amt = result.get("filled_amount", coin_amt)
            price = result.get("avg_price", price)
            ledger.setdefault("trades", []).append(
                {
                    "action": "buy",
                    "symbol": tag,
                    "usd": args.usd,
                    "coin": coin_amt,
                    "price": price,
                    "timestamp": result.get("timestamp"),
                }
            )
            save_ledger(args.ledger, ledger)
        addlog(
            f"[MANUAL BUY] {args.ledger} | {tag} | ${args.usd:.2f} → {coin_amt:.4f} {coin_str} @ ${price:.4f}",
            verbose_int=1,
            verbose_state=args.verbose,
        )
    else:  # sell
        if not args.dry:
            result = execute_sell(
                None,
                pair_code=ledger_cfg["kraken_pair"],
                coin_amount=coin_amt,
                price=price,
                ledger_name=args.ledger,
                verbose=args.verbose,
            )
            if not result or result.get("error"):
                raise SystemExit("[ERROR] Sell order failed")
            coin_amt = result.get("filled_amount", coin_amt)
            price = result.get("avg_price", price)
            usd_total = coin_amt * price
            ledger.setdefault("trades", []).append(
                {
                    "action": "sell",
                    "symbol": tag,
                    "usd": usd_total,
                    "coin": coin_amt,
                    "price": price,
                    "timestamp": result.get("timestamp"),
                }
            )
            save_ledger(args.ledger, ledger)
        else:
            usd_total = args.usd
        addlog(
            f"[MANUAL SELL] {args.ledger} | {tag} | {coin_amt:.4f} {coin_str} → ${usd_total:.2f} @ ${price:.4f}",
            verbose_int=1,
            verbose_state=args.verbose,
        )


if __name__ == "__main__":
    main()
