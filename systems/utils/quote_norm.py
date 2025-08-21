NORM_MAP = {
    "ZUSD": "USD",
    "ZEUR": "EUR",
    "ZGBP": "GBP",
    "ZJPY": "JPY",
}

def norm_quote(q: str) -> str:
    q = (q or "").upper()
    return NORM_MAP.get(q, q)


def assert_quote_match(*, quote_expected: str, exchange_pair: str) -> None:
    """Ensure ``exchange_pair`` ends with ``quote_expected`` (after normalization)."""
    try:
        _, q = exchange_pair.split("/")
    except ValueError as exc:
        raise ValueError(f"Invalid market symbol '{exchange_pair}'") from exc
    if norm_quote(q) != norm_quote(quote_expected):
        raise ValueError(
            f"Quote mismatch: expected {quote_expected} for pair {exchange_pair}"
        )
