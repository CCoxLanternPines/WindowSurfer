"""Run a series of manual configuration trials without Optuna.

This module provides a lightweight alternative to the broken Optuna tuner.
Configurations are defined inline and injected into the simulation settings
one-by-one.  After each simulation the resulting PnL and capital usage are
recorded to ``data/tmp/manual_leaderboard.csv`` for inspection.

Usage
-----
``python -m systems.manual_tune --tag DOGEUSD -v``
"""

from __future__ import annotations

import argparse
import json
from typing import Any, Dict, List

import csv

from systems.sim_engine import run_simulation
from systems.utils.path import find_project_root


# ---------------------------------------------------------------------------
# Test configurations
# ---------------------------------------------------------------------------

test_configs: List[Dict[str, Any]] = [
    {
        "buy_floor": 0.1,
        "sell_ceiling": 0.9,
        "investment_fraction": 0.1,
        "cooldown": 6,
        "window_size": "3d",
    },
    {
        "buy_floor": 0.2,
        "sell_ceiling": 0.85,
        "investment_fraction": 0.15,
        "cooldown": 12,
        "window_size": "3d",
    },
    {
        "buy_floor": 0.3,
        "sell_ceiling": 0.8,
        "investment_fraction": 0.2,
        "cooldown": 24,
        "window_size": "3d",
    },
]


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------


def _load_base_settings() -> Dict[str, Any]:
    """Load ``settings.json`` from the project settings directory."""

    root = find_project_root()
    path = root / "settings" / "settings.json"
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _compute_capital_used(notes: List[Dict[str, Any]]) -> float:
    """Return peak capital committed across all notes."""

    events: List[tuple[int, float]] = []
    for note in notes:
        entry = int(note.get("entry_tick", 0))
        exit_ = note.get("exit_tick")
        exit_tick = int(exit_) if exit_ is not None else entry + 10**9
        amt = float(note.get("entry_usdt", 0.0))
        events.append((entry, amt))
        events.append((exit_tick, -amt))

    active = 0.0
    peak = 0.0
    for _, delta in sorted(events, key=lambda x: x[0]):
        active += delta
        peak = max(peak, active)
    return peak


# ---------------------------------------------------------------------------
# Core tuning routine
# ---------------------------------------------------------------------------


def run_trials(tag: str, verbose: int = 0) -> None:
    """Execute all test configurations for ``tag``."""

    base_settings = _load_base_settings()
    root = find_project_root()
    tmp_dir = root / "data" / "tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    leaderboard_path = tmp_dir / "manual_leaderboard.csv"

    records: List[Dict[str, Any]] = []

    for idx, cfg in enumerate(test_configs, start=1):
        settings = json.loads(json.dumps(base_settings))
        settings.setdefault("general_settings", {}).setdefault("windows", {})[
            "fish"
        ] = cfg

        run_simulation(tag=tag, settings=settings, verbose=0)

        ledger_path = tmp_dir / "ledgersimulation.json"
        with ledger_path.open("r", encoding="utf-8") as f:
            ledger = json.load(f)

        pnl = float(ledger.get("pnl", 0.0))
        notes = ledger.get("open_notes", []) + ledger.get("closed_notes", [])
        capital_used = _compute_capital_used(notes)
        score = pnl / capital_used if capital_used > 0 else 0.0

        record = {"score": score, "pnl": pnl, "capital_used": capital_used, **cfg}
        records.append(record)

        if verbose:
            print(
                f"[TRIAL {idx}] Score={score:.2f} | PnL=${pnl:.2f} | "
                f"Cap=${capital_used:.2f} | Config={cfg}"
            )

    sorted_records = sorted(records, key=lambda r: r["score"], reverse=True)
    if sorted_records:
        fieldnames = list(sorted_records[0].keys())
        with leaderboard_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(sorted_records)

        print("\n🏆 Top 5:")
        print(",".join(fieldnames))
        for row in sorted_records[:5]:
            print(",".join(str(row[k]) for k in fieldnames))
    else:
        with leaderboard_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["score", "pnl", "capital_used"])
            writer.writeheader()
        print("\n🏆 Top 5:\n( no records )")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Run manual configuration trials")
    parser.add_argument("--tag", required=True, help="Symbol tag, e.g. DOGEUSD")
    parser.add_argument("-v", dest="verbose", action="count", default=0)
    args = parser.parse_args()

    run_trials(tag=args.tag.upper(), verbose=args.verbose)


if __name__ == "__main__":
    main()

