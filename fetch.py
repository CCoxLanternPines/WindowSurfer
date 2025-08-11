import argparse

from systems.scripts.fetch_core import (
    build_wallet_cache,
    fetch_full_history,
    fetch_update_history,
)
from systems.scripts.config_loader import load_runtime_config


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
        cfg = load_runtime_config(args.ledger, runtime_mode="fetch")
        fiat = cfg.get("fiat", "USD")
        coins_cfg = cfg.get("coins", {})
        for symbol in coins_cfg.keys():
            if args.full:
                df = fetch_full_history(symbol, fiat)
                print(f"[FETCH] {symbol}: fetched {len(df)} candles from Binance")
            if args.update:
                fetch_update_history(symbol, fiat)
        if args.full:
            coin_list = ", ".join(coins_cfg.keys())
            print(f"[FETCH] Full history fetch complete for ledger '{args.ledger}' ✅")
            print(f"[FETCH] Coins fetched: {coin_list}")


if __name__ == "__main__":
    main()
