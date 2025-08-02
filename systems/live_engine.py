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
from systems.utils.logger import addlog


def run_live(
    tag: str | None = None,
    window: str | None = None,
    dry: bool = False,
    verbose: int = 0,
) -> None:
    """Run the live trading engine.

    Parameters are currently placeholders for forward compatibility.
    """
    settings = load_settings()
    tick_time = datetime.now(timezone.utc)

    if dry:
        for ledger_key, ledger_cfg in settings.get("ledger_settings", {}).items():
            tag = ledger_cfg.get("tag")
            fetch_missing_candles(tag, relative_window="48h", verbose=verbose)
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


def _parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Live trading engine")
    parser.add_argument("--dry", action="store_true", help="Run once immediately")
    parser.add_argument("--tag", required=False, help="Symbol tag (unused)")
    parser.add_argument("--window", required=False, help="Window name (unused)")
    return parser.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> None:
    args = _parse_args(argv)
    run_live(tag=args.tag, window=args.window, dry=args.dry)


if __name__ == "__main__":
    main()
