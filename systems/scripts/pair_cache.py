from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import ccxt


def _group_markets_by_base(markets: Dict[str, Dict]) -> Dict[str, List[Dict]]:
    grouped: Dict[str, List[Dict]] = {}
    for m in markets.values():
        base = m.get("base")
        quote = m.get("quote")
        if not base or not quote:
            continue
        base = str(base).upper()
        grouped.setdefault(base, []).append(m)
    return grouped


def _filter_quotes(markets: List[Dict], preferred: Tuple[str, ...]) -> List[Dict]:
    preferred_set = {q.upper() for q in preferred}
    selected = [m for m in markets if str(m.get("quote", "")).upper() in preferred_set]
    return selected if selected else list(markets)


def _sort_by_quote_priority(markets: List[Dict], preferred: Tuple[str, ...]) -> List[Dict]:
    def key(m: Dict) -> Tuple[int, str]:
        quote = str(m.get("quote", "")).upper()
        try:
            idx = preferred.index(quote)
        except ValueError:
            idx = len(preferred)
        return (idx, quote)

    return sorted(markets, key=key)


def update_pair_cache(
    out_path: str = "data/tmp/pair_cache.json",
    preferred_quotes: Tuple[str, ...] = ("USD", "USDT", "USDC"),
) -> Dict[str, Dict]:
    """Refresh pair cache for Kraken and Binance and persist to ``out_path``."""

    kr = ccxt.kraken({"enableRateLimit": True})
    bn = ccxt.binance({"enableRateLimit": True})

    kr_markets = kr.load_markets()
    bn_markets = bn.load_markets()

    kr_by_base = _group_markets_by_base(kr_markets)
    bn_by_base = _group_markets_by_base(bn_markets)

    bases = sorted(set(kr_by_base) | set(bn_by_base))
    cache: Dict[str, Dict] = {}

    for base in bases:
        bn_markets_base = _filter_quotes(bn_by_base.get(base, []), preferred_quotes)
        kr_markets_base = _filter_quotes(kr_by_base.get(base, []), preferred_quotes)

        bn_markets_base = _sort_by_quote_priority(bn_markets_base, preferred_quotes)
        kr_markets_base = _sort_by_quote_priority(kr_markets_base, preferred_quotes)

        binance_entries: List[Dict] = []
        for m in bn_markets_base:
            entry = {
                "symbol": m.get("id"),
                "base": m.get("base"),
                "quote": m.get("quote"),
                "id": m.get("symbol"),
                "active": m.get("active", True),
            }
            binance_entries.append(entry)

        kraken_entries: List[Dict] = []
        for m in kr_markets_base:
            info = m.get("info", {})
            base_id = m.get("baseId")
            quote_id = m.get("quoteId")
            pair = f"{base_id}{quote_id}" if base_id and quote_id else None
            entry = {
                "symbol": m.get("symbol"),
                "altname": info.get("altname"),
                "wsname": info.get("wsname"),
                "baseId": base_id,
                "quoteId": quote_id,
                "pair": pair,
                "active": m.get("active", True),
            }
            kraken_entries.append(entry)

        wallet_codes: Dict[str, Optional[str]] = {}
        if kraken_entries:
            wallet_codes["kraken_asset"] = kraken_entries[0].get("baseId")
        if binance_entries:
            wallet_codes["binance_asset"] = binance_entries[0].get("base")

        cache[base] = {
            "binance": binance_entries,
            "kraken": kraken_entries,
            "wallet_codes": wallet_codes,
        }

    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(cache, f, indent=2, sort_keys=True)

    return cache


def lookup_coin(
    coin: str,
    cache: Optional[Dict[str, Dict]] = None,
    cache_path: str = "data/tmp/pair_cache.json",
) -> Optional[Dict]:
    """Return cache entry for ``coin`` (case-insensitive) or ``None``."""

    if not coin:
        return None
    normalized = coin.strip().upper()

    if cache is None:
        path = Path(cache_path)
        if not path.exists():
            return None
        with path.open("r", encoding="utf-8") as f:
            cache = json.load(f)

    base = normalized
    if "/" in base:
        base = base.split("/")[0]
    else:
        quotes: set[str] = set()
        for data in cache.values():
            for m in data.get("binance", []):
                q = m.get("quote")
                if q:
                    quotes.add(q.upper())
            for m in data.get("kraken", []):
                sym = m.get("symbol")
                if sym and "/" in sym:
                    quotes.add(sym.split("/")[1].upper())
        for q in sorted(quotes, key=len, reverse=True):
            if base.endswith(q):
                base = base[: -len(q)]
                break

    return cache.get(base)
