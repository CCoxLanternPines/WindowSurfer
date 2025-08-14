from __future__ import annotations

"""CLI helpers for candle fetching."""

import sys
from pathlib import Path

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from systems.utils.addlog import addlog
from systems.utils.config import load_settings
from systems.scripts.candle_refresh import refresh_to_last_closed_hour


def _parse_lookback(value: str) -> int:
    value = value.strip().lower()
    if value.endswith("h"):
        value = value[:-1]
    return int(value)


def run_fetch(ledger: str | None, *, lookback: str | None) -> None:
    """Fetch missing candles for ``ledger`` within ``lookback`` hours."""

    if not ledger:
        addlog(
            "Error: --ledger is required for fetch mode",
            verbose_int=1,
            verbose_state=True,
        )
        raise SystemExit(1)

    if not lookback:
        addlog(
            "Error: --time is required for fetch mode",
            verbose_int=1,
            verbose_state=True,
        )
        raise SystemExit(1)

    hours = _parse_lookback(lookback)
    settings = load_settings()
    ledger_cfg = settings.get("ledger_settings", {}).get(ledger)
    if not ledger_cfg:
        addlog(
            f"Error: Unknown ledger {ledger}",
            verbose_int=1,
            verbose_state=True,
        )
        raise SystemExit(1)

    tag = ledger_cfg.get("tag")
    addlog(
        f"[BOT][FETCH] ledger={ledger} tag={tag} lookback={hours}h",
        verbose_int=1,
        verbose_state=True,
    )
    refresh_to_last_closed_hour(settings, tag, lookback_hours=hours, exchange="kraken")

