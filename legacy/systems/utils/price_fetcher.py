import json
import urllib.request


def get_price(symbol: str) -> float | None:
    """Fetch the latest close price for a trading pair from Kraken."""
    url = f"https://api.kraken.com/0/public/Ticker?pair={symbol}"
    try:
        with urllib.request.urlopen(url) as response:
            data = json.load(response)
            result = list(data.get("result", {}).values())
            if not result:
                return None
            return float(result[0]["c"][0])
    except Exception:
        return None
