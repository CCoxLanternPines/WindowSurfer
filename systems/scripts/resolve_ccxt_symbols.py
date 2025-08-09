def resolve_ccxt_symbols(ledger_cfg: dict) -> dict:
    return {
        "kraken": ledger_cfg["kraken_name"],
        "binance": ledger_cfg["binance_name"],
    }
