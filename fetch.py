import argparse
import json
from pathlib import Path

from systems.scripts.fetch_core import build_wallet_cache, fetch_full_history, fetch_update_history


def _load_ledger(name: str):
    path = Path("data/ledgers") / f"{name}.json"
    if not path.exists():
        raise FileNotFoundError(f"ledger not found: {path}")
    with open(path, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    if isinstance(data, dict):
        return list(data.items())
    coins = []
    for entry in data:
        symbol = entry.get("symbol")
        fiat = entry.get("fiat")
        if symbol and fiat:
            coins.append((symbol, fiat))
    return coins


def main() -> None:
    parser = argparse.ArgumentParser(description="Unified market data fetch commands")
    parser.add_argument("--ledger", help="Ledger name for coin list")
    parser.add_argument("--full", action="store_true", help="Fetch full Binance history")
    parser.add_argument("--update", action="store_true", help="Append recent Kraken candles")
    parser.add_argument("--wallet_cache", action="store_true", help="Update wallet cache files")
    args = parser.parse_args()

    if args.wallet_cache:
        build_wallet_cache()

    if args.full or args.update:
        if not args.ledger:
            parser.error("--ledger is required for full or update fetch")
        coins = _load_ledger(args.ledger)
        for symbol, fiat in coins:
            if args.full:
                fetch_full_history(symbol, fiat)
            if args.update:
                fetch_update_history(symbol, fiat)


if __name__ == "__main__":
    main()
