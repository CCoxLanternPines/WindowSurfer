from __future__ import annotations

"""Live trading entry engine."""

import argparse
import time
from datetime import datetime, timezone
from typing import Optional

from tqdm import tqdm

from systems.scripts.handle_top_of_hour import handle_top_of_hour
from systems.utils.settings_loader import load_settings
from systems.fetch import fetch_missing_candles
from systems.utils.addlog import addlog
from systems.utils.resolve_symbol import resolve_ledger_settings


def run_live(
    ledger_name: str,
    window: str | None = None,
    dry: bool = False,
    verbose: int = 0,
) -> None:
    """Run the live trading engine for ``ledger_name``."""
    settings = load_settings()
    ledger_cfg = resolve_ledger_settings(ledger_name, settings)
    kraken_pair = ledger_cfg.get("kraken_pair")
    wallet_code = ledger_cfg.get("wallet_code")
    fiat_code = ledger_cfg.get("fiat")
    _ = kraken_pair, wallet_code, fiat_code  # Ensure variables are resolved
    tick_time = datetime.now(timezone.utc)

    def _run_top_of_hour(ts: datetime) -> None:
        handle_top_of_hour(
            tick=ts,
            ledger_name=ledger_name,
            settings=settings,
            sim=False,
            dry=dry,
            verbose=verbose,
        )

    if dry:
        tag = ledger_cfg.get("tag")
        fetch_missing_candles(ledger_name, relative_window="48h", verbose=verbose)
        addlog(
            f"[SYNC] {ledger_name} | {tag} candles up to date",
            verbose_int=1,
            verbose_state=verbose,
        )
        addlog("[LIVE] Running top of hour", verbose_int=1, verbose_state=verbose)
        _run_top_of_hour(tick_time)
        return

    while True:
        now = datetime.utcnow().replace(tzinfo=timezone.utc)
        elapsed_secs = now.minute * 60 + now.second
        remaining_secs = 3600 - elapsed_secs

        with tqdm(
            total=3600,
            initial=elapsed_secs,
            desc="⏳ Time to next hour",
            bar_format="{l_bar}{bar}| {percentage:3.0f}% {remaining}s",
            leave=True,
            dynamic_ncols=True,
        ) as pbar:
            for _ in range(remaining_secs):
                time.sleep(1)
                pbar.update(1)

        addlog("[LIVE] Running top of hour", verbose_int=1, verbose_state=verbose)
        _run_top_of_hour(datetime.now(timezone.utc))


def _parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Live trading engine")
    parser.add_argument("--dry", action="store_true", help="Run once immediately")
    parser.add_argument("--ledger", required=True, help="Ledger name")
    parser.add_argument("--window", required=False, help="Window name (unused)")
    return parser.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> None:
    args = _parse_args(argv)
    run_live(ledger_name=args.ledger, window=args.window, dry=args.dry)


if __name__ == "__main__":
    main()
