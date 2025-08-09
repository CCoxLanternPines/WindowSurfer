import json
from pathlib import Path

CACHE_PATH = Path("data/tmp/pair_cache.json")


class PairNotFound(Exception):
    pass


def load_cache(path: str | Path = CACHE_PATH) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def resolve_by_tag(tag: str, cache: dict | None = None) -> dict:
    """
    Returns a dict of canonical symbols/codes for the logical tag.
    Expected cache structure: cache[tag] exists.
    Output (minimum):
      {
        "binance_symbol": "DOGEUSDT",
        "kraken_symbol": "DOGE/USD",
        "kraken_wallet_code": "XXDG",   # if present in cache.wallet_codes
        "binance_base": "DOGE",         # convenience
      }
    Selection rules:
      - Choose the first Binance market for this base with a quote in priority [USD, USDT, USDC];
        fallback to any if none match.
      - Choose the first Kraken market similarly; fallback to any if none match.
    """
    cache = cache or load_cache()
    entry = cache.get(tag.upper())
    if not entry:
        raise PairNotFound(f"No pair info for tag '{tag}' in {CACHE_PATH}")
    # Binance pick
    b_list = entry.get("binance") or []
    b_pick = _pick_by_quotes(b_list, ("USD", "USDT", "USDC"), "symbol")
    # Kraken pick
    k_list = entry.get("kraken") or []
    k_pick = _pick_by_quotes(k_list, ("USD", "USDT", "USDC"), "symbol")
    # Wallet codes if present
    wc = (entry.get("wallet_codes") or {})
    return {
        "binance_symbol": (b_pick or {}).get("symbol"),
        "kraken_symbol": (k_pick or {}).get("symbol"),
        "kraken_wallet_code": wc.get("kraken_asset"),
        "binance_base": (b_pick or {}).get("base")
        or wc.get("binance_asset")
        or tag.upper(),
    }


def _pick_by_quotes(markets: list[dict], priority: tuple[str, ...], symbol_key: str) -> dict | None:
    # markets contain unified keys from ccxt: 'symbol', 'base', 'quote'
    by_quote = {m.get("quote"): m for m in markets if m.get(symbol_key)}
    for q in priority:
        if q in by_quote:
            return by_quote[q]
    return markets[0] if markets else None
