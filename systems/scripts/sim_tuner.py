from __future__ import annotations

"""Run per-window simulations and export results for tuning."""

import csv
from typing import Any

from systems.sim_engine import run_simulation
from systems.scripts.fetch_canles import fetch_candles
from systems.scripts.ledger import Ledger
from systems.utils.addlog import addlog
from systems.utils.config import load_ledger_config
from systems.utils.path import find_project_root


def run_sim_tuner(*, ledger: str, verbose: int = 0) -> None:
    """Simulate each window for ``ledger`` and export results to CSV."""

    ledger_cfg = load_ledger_config(ledger)
    tag = ledger_cfg.get("tag", "")
    window_settings = ledger_cfg.get("window_settings", {})
    if not window_settings:
        raise ValueError("No windows defined for ledger")

    root = find_project_root()
    final_price = float(fetch_candles(tag).iloc[-1]["close"])

    for window_name, window_cfg in window_settings.items():
        addlog(
            f"[SIMTUNE] Running {ledger} | {window_name} window",
            verbose_int=1,
            verbose_state=verbose,
        )
        run_simulation(ledger=ledger, verbose=verbose, window_names=[window_name])
        ledger_obj = Ledger.load_ledger(ledger, sim=True)
        summary = ledger_obj.get_account_summary(final_price)

        csv_path = root / "data" / "tmp" / f"{ledger}_{window_name}_{tag}.csv"
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = ["window", *window_cfg.keys(), "closed_notes", "realized_gain", "total_value"]
        row: dict[str, Any] = {
            "window": window_name,
            **window_cfg,
            "closed_notes": summary.get("closed_notes"),
            "realized_gain": summary.get("realized_gain"),
            "total_value": summary.get("total_value"),
        }
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerow(row)
        addlog(
            f"[SIMTUNE] Saved results to {csv_path}",
            verbose_int=1,
            verbose_state=verbose,
        )
