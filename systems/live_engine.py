from __future__ import annotations

import argparse

from systems.utils.config import load_ledgers, resolve_ledger_cfg
from systems.utils.pairs import resolve_by_tag


def run(args: argparse.Namespace) -> None:
    all_cfg = load_ledgers("settings/ledgers.json")
    ledger_name = args.ledger or next(iter(all_cfg.get("ledgers", {})))
    cfg = resolve_ledger_cfg(ledger_name, all_cfg)

    if "tag" not in cfg:
        raise SystemExit(f"[error] Ledger '{ledger_name}' missing 'tag' in settings/ledgers.json")

    pair = resolve_by_tag(cfg["tag"])
    kraken_wallet_code = pair.get("kraken_wallet_code")
    kraken_symbol = pair["kraken_symbol"]
    binance_symbol = pair["binance_symbol"]

    window_size = cfg["window_size"]
    investment_size = float(cfg["investment_size"])
    buy_multiplier = float(cfg["buy_multiplier"])
    sell_multiplier = float(cfg["sell_multiplier"])

    wa = cfg["window_alg"]
    window_alg_window = wa["window"]
    window_alg_skip_candles = int(wa["skip_candles"])

    TUN = cfg["tunnel_settings"]
    alpha_wick = float(TUN["alpha_wick"])
    smooth_ema = float(TUN["smooth_ema"])
    momentum_bars = int(TUN["momentum_bars"])
    momentum_eps = float(TUN["momentum_eps"])
    dead_zone_pct = TUN.get("dead_zone_pct")
    dead_zone_min = float(TUN["dead_zone_min"])
    dead_zone_max = float(TUN["dead_zone_max"])

    snapback_lookback = int(cfg["snapback_odds"]["lookback"])

    print(
        f"[LIVE] Loaded {ledger_name} | Kraken:{kraken_symbol} Binance:{binance_symbol} Window:{window_size}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Live trading engine stub.")
    parser.add_argument("--ledger", dest="ledger", help="Ledger name to use")
    parser.add_argument("--dry", action="store_true", help="Dry run")
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
