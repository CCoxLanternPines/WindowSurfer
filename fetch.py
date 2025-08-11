import argparse
import json

from systems.scripts.fetch_core import (
    build_wallet_cache,
    fetch_full_history,
    fetch_update_history,
)
from systems.scripts.path_utils import ledger_settings_path


def _load_ledger(name: str):
    path = ledger_settings_path(name)
    if not path.exists():
        raise FileNotFoundError(f"ledger settings not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        ledger_data = json.load(f)
    fiat = ledger_data.get("fiat")
    coins = [{"symbol": sym, "fiat": fiat} for sym in ledger_data["coins"].keys()]
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
            print("[ERR] --ledger is required for this command and exit.")
            return
        coins = _load_ledger(args.ledger)
        for coin in coins:
            symbol = coin["symbol"]
            fiat = coin["fiat"]
            if args.full:
                fetch_full_history(symbol, fiat)
            if args.update:
                fetch_update_history(symbol, fiat)


if __name__ == "__main__":
    main()
