from __future__ import annotations

"""Live trading entry engine."""

import time
from datetime import datetime, timezone
from typing import Optional

from tqdm import tqdm

from systems.scripts.handle_top_of_hour import handle_top_of_hour
from systems.utils.config import load_settings
from systems.fetch import fetch_missing_candles
from systems.utils.addlog import addlog
from systems.utils.cli import build_parser


def run_live(*, dry: bool = False, verbose: int = 0) -> None:
    """Run the live trading engine."""
    settings = load_settings()
    tick_time = datetime.now(timezone.utc)

    if dry:
        for ledger_key, ledger_cfg in settings.get("ledger_settings", {}).items():
            tag = ledger_cfg.get("tag")
            fetch_missing_candles(ledger_key, relative_window="48h", verbose=verbose)
            addlog(
                f"[SYNC] {ledger_key} | {tag} candles up to date",
                verbose_int=1,
                verbose_state=verbose,
            )
        addlog("[LIVE] Running top of hour", verbose_int=1, verbose_state=verbose)
        handle_top_of_hour(
            tick=tick_time,
            settings=settings,
            sim=False,
            dry=dry,
            verbose=verbose,
        )
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
        handle_top_of_hour(
            tick=datetime.now(timezone.utc),
            settings=settings,
            sim=False,
            dry=dry,
            verbose=verbose,
        )


def _parse_args(argv: Optional[list[str]] = None):
    parser = build_parser()
    return parser.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> None:
    args = _parse_args(argv)
    run_live(dry=args.dry)


if __name__ == "__main__":
    main()
