from __future__ import annotations

import csv
import json
import os
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict

import optuna

from sim_engine import run_sim


def run(trials: int) -> None:
    """Run an Optuna study for the provided number of ``trials``."""
    with open("settings.json", "r", encoding="utf-8") as f:
        base_cfg: Dict[str, Any] = json.load(f)
    with open("knobs.json", "r", encoding="utf-8") as f:
        knobs: Dict[str, list[float]] = json.load(f)

    results_path = Path("tune_results.csv")

    def objective(trial: optuna.trial.Trial) -> float:
        cfg = base_cfg.copy()
        params: Dict[str, float] = {}
        for name, bounds in knobs.items():
            low, high = bounds
            if isinstance(low, int) and isinstance(high, int):
                value = trial.suggest_int(name, int(low), int(high))
            else:
                value = trial.suggest_float(name, float(low), float(high))
            cfg[name] = value
            params[name] = value

        metrics = run_sim(cfg)

        row = OrderedDict([("trial", trial.number)])
        for k in knobs.keys():
            row[k] = params[k]
        row["final_capital"] = metrics["final_capital"]
        row["pnl"] = metrics["pnl"]

        fieldnames = list(row.keys())
        header_line = ",".join(fieldnames)
        if results_path.exists():
            existing_content = results_path.read_text(encoding="utf-8")
            first_line = existing_content.splitlines()[0] if existing_content else ""
            if first_line != header_line:
                with results_path.open("w", newline="", encoding="utf-8") as f:
                    writer = csv.writer(f)
                    writer.writerow(fieldnames)
                    if existing_content:
                        f.write(existing_content)
        else:
            with results_path.open("w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(fieldnames)

        with results_path.open("a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writerow(row)
            f.flush()
            os.fsync(f.fileno())

        return float(metrics["pnl"])

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=trials)
