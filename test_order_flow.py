import time
import argparse

from systems.utils.resolve_symbol import resolve_symbol
from systems.scripts.execution_handler import buy_order, sell_order
from systems.scripts.kraken_auth import load_kraken_keys


def main() -> None:
    parser = argparse.ArgumentParser(description="Execute live buy then sell on Kraken")
    parser.add_argument("symbol", help="Symbol tag e.g. DOGEUSD")
    parser.add_argument("amount", type=float, help="Amount in USD to trade")
    parser.add_argument("-v", "--verbose", action="count", default=0, help="Increase verbosity")
    args = parser.parse_args()

    # Ensure credentials load before prompting
    try:
        load_kraken_keys()
    except Exception as exc:
        print(f"[ERROR] Unable to load kraken credentials: {exc}")
        return

    resolved = resolve_symbol(args.symbol)
    kraken_symbol = resolved["kraken"]
    usd_amount = args.amount

    print(f"[RESOLVED] Kraken symbol: {kraken_symbol}")
    print(f"[TEST] About to BUY ${usd_amount} of {kraken_symbol} on Kraken")
    confirm = input("Are you sure? (y/n): ").strip().lower()
    if confirm != "y":
        print("Aborting.")
        return

    try:
        buy_resp = buy_order(args.symbol, usd_amount, verbose=args.verbose)
        print(f"[BUY EXECUTED] Price: {buy_resp.get('price')} | Amount Filled: {buy_resp.get('volume')}")
        print(buy_resp)
    except Exception as exc:
        print(f"[BUY ERROR] {exc}")
        return

    print("[SLEEP] Waiting 5 seconds...")
    time.sleep(5)

    try:
        sell_resp = sell_order(args.symbol, usd_amount, verbose=args.verbose)
        print(f"[SELL EXECUTED] Price: {sell_resp.get('price')} | Amount Sold: {sell_resp.get('volume')}")
        print(sell_resp)
    except Exception as exc:
        print(f"[SELL ERROR] {exc}")
        return


if __name__ == "__main__":
    main()
