"""Command line entry point for WindowSurfer."""

from __future__ import annotations

import sys

from systems.live_engine import run_live
from systems.sim_engine import run_simulation
from systems.utils.addlog import init_logger, addlog
from systems.utils.config import load_settings, load_ledger_config
from systems.utils.resolve_symbol import load_pair_cache, resolve_ccxt_symbols
from systems.utils.cli import build_parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    for action in parser._actions:
        if action.dest == "mode" and action.choices is not None:
            action.choices = list(action.choices) + ["fetch"]
            break
    parser.add_argument("--time", required=False, help="Time window (e.g. 120h)")
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Perform a paper buy+sell if no actions occur (dry live mode)",
    )
    parser.add_argument(
        "--smoke-save",
        action="store_true",
        help="Persist smoke ledger to data/tmp/",
    )
    parser.add_argument(
        "--replay-hours",
        type=int,
        help="After dry tick, replay the last N hours in sim mode",
    )
    parser.add_argument(
        "--cache",
        action="store_true",
        help="Refresh exchange pair cache before running",
    )

    args = parser.parse_args(argv or sys.argv[1:])
    if not args.mode:
        parser.error("--mode is required")
    init_logger(
        logging_enabled=args.log,
        verbose_level=args.verbose,
        telegram_enabled=args.telegram,
    )

    mode = args.mode.lower()
    verbose = args.verbose

    settings = load_settings()

    try:
        cache = load_pair_cache()
    except Exception:
        addlog(
            "[ERROR] Failed to load pair cache",
            verbose_int=1,
            verbose_state=True,
        )
        sys.exit(1)

    for ledger_cfg in settings.get("ledger_settings", {}).values():
        coin = ledger_cfg.get("coin")
        fiat = ledger_cfg.get("fiat")
        try:
            resolve_ccxt_symbols(coin, fiat, cache, verbose)
        except Exception as e:
            raise RuntimeError(
                f"[ERROR] Invalid trading pair: {coin}/{fiat} — {e}"
            )

    if mode == "wallet":
        from systems.scripts.kraken_utils import get_kraken_balance

        if args.ledger:
            ledger_cfg = load_ledger_config(args.ledger)
        else:
            ledger_cfg = next(iter(settings.get("ledger_settings", {}).values()))

        coin = ledger_cfg["coin"]
        fiat = ledger_cfg["fiat"]
        balances = get_kraken_balance(fiat, verbose)

        if verbose >= 1:
            addlog("[WALLET] Kraken Balance", verbose_int=1, verbose_state=verbose)
            addlog(str(balances), verbose_int=2, verbose_state=verbose)
            for asset, amount in balances.items():
                val = float(amount)
                if val == 0:
                    continue
                fmt = f"{val:.2f}" if val > 1 else f"{val:.6f}"
                if asset.upper() == fiat.upper():
                    addlog(f"{asset}: ${fmt}", verbose_int=1, verbose_state=verbose)
                else:
                    addlog(f"{asset}: {fmt}", verbose_int=1, verbose_state=verbose)
        return

    if mode == "sim":
        if not args.ledger:
            addlog("Error: --ledger is required for sim mode")
            sys.exit(1)
        run_simulation(ledger=args.ledger, verbose=args.verbose)
    elif mode == "simtune":
        if not args.ledger:
            addlog("Error: --ledger is required for simtune mode")
            sys.exit(1)
        from systems.scripts.sim_tuner import run_sim_tuner
        run_sim_tuner(ledger=args.ledger, verbose=args.verbose)
    elif mode == "live":
        run_live(
            dry=args.dry,
            verbose=args.verbose,
            smoke=args.smoke,
            smoke_save=args.smoke_save,
            replay_hours=args.replay_hours,
        )
    elif mode == "fetch":
        if not args.ledger:
            addlog("Error: --ledger is required for fetch mode", verbose_int=1, verbose_state=verbose)
            sys.exit(1)
        time_window = args.time if args.time else "48h"
        try:
            from systems.fetch import fetch_missing_candles

            fetch_missing_candles(
                ledger=args.ledger,
                relative_window=time_window,
                verbose=args.verbose,
                refresh_cache=args.cache,
            )
        except Exception as e:
            addlog(f"[ERROR] Fetch failed: {e}", verbose_int=1, verbose_state=True)
            sys.exit(1)
        sys.exit(0)
    else:
        addlog(
            "Error: --mode must be either 'sim', 'simtune', 'live', 'wallet', or 'fetch'",
            verbose_int=1,
            verbose_state=verbose,
        )
        sys.exit(1)


if __name__ == "__main__":
    main()
