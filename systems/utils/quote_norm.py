NORM_MAP = {
    "ZUSD": "USD",
    "ZEUR": "EUR",
    "ZGBP": "GBP",
    "ZJPY": "JPY",
}

def norm_quote(q: str) -> str:
    q = (q or "").upper()
    return NORM_MAP.get(q, q)
