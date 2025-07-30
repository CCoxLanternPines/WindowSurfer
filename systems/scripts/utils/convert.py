from __future__ import annotations

import json
import sys
import urllib.request


def get_price(symbol: str = "DOGEUSD") -> float | None:
    url = f"https://api.kraken.com/0/public/Ticker?pair={symbol}"
    try:
        with urllib.request.urlopen(url) as response:
            data = json.load(response)
            result = list(data["result"].values())[0]
            return float(result["c"][0])
    except Exception as e:
        print(f"[ERROR] Failed to fetch price for {symbol}: {e}")
        return None


def main() -> None:
    if len(sys.argv) != 3:
        print("Usage: python convert.py SYMBOL AMOUNT")
        print("Example: python convert.py DOGEUSD 1250")
        return

    symbol = sys.argv[1]
    amount = float(sys.argv[2])
    price = get_price(symbol)

    if price:
        value = price * amount
        print(f"{amount} {symbol[:-3]} \u2248 ${value:.2f} USD")


if __name__ == "__main__":
    main()
