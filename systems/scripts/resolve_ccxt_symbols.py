from systems.utils.pairs import resolve_by_tag


def resolve_ccxt_symbols(ledger_cfg: dict) -> dict:
    # ledger_cfg MUST have 'tag'
    info = resolve_by_tag(ledger_cfg["tag"])
    return {
        "kraken": info["kraken_symbol"],
        "binance": info["binance_symbol"],
    }
