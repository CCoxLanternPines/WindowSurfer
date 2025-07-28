import time
import argparse
from systems.scripts.execution_handler import buy_order, sell_order
from systems.utils.resolve_symbol import resolve_symbol
from systems.utils.logger import addlog

def main():
    parser = argparse.ArgumentParser(description="Test Kraken Buy/Sell Roundtrip")
    parser.add_argument("--tag", type=str, required=True, help="Trading pair (e.g., DOGEUSD)")
    parser.add_argument("--usd", type=float, default=5.0, help="USD amount to trade")
    parser.add_argument("-v", "--verbose", action="count", default=0, help="Increase verbosity (use -v, -vv)")
    args = parser.parse_args()

    symbol = args.tag.upper()
    usd_amount = args.usd
    verbose = args.verbose  # <-- FIXED HERE

    kraken_pair = resolve_symbol(symbol)["kraken"]
    addlog(
        f"[INFO] Resolved {symbol} → Kraken pair {kraken_pair}",
        verbose_int=1,
        verbose_state=verbose,
    )

    addlog(f"--- BUYING ${usd_amount:.2f} of {symbol} ---")
    buy_result = buy_order(symbol, usd_amount, verbose=verbose)
    addlog(f"[✓] Buy Complete: {buy_result}")

    addlog("\n⏳ Waiting 5 seconds before selling...\n")
    time.sleep(5)

    addlog(f"--- SELLING ${usd_amount:.2f} of {symbol} ---")
    sell_result = sell_order(symbol, usd_amount, verbose=verbose)
    addlog(f"[✓] Sell Complete: {sell_result}")

if __name__ == "__main__":
    main()
