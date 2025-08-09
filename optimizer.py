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
        knobs: Dict[str, list[int]] = json.load(f)

    results_path = Path("tune_results.csv")

    def objective(trial: optuna.trial.Trial) -> float:
        cfg = base_cfg.copy()
        params: Dict[str, int] = {}
        for name, bounds in knobs.items():
            low, high = bounds
            value = trial.suggest_int(name, int(low), int(high))
            cfg[name] = value
            params[name] = value

        metrics = run_sim(cfg)

        row = OrderedDict([("trial", trial.number)])
        row.update(params)
        row.update(metrics)

        file_exists = results_path.exists()
        with results_path.open("a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=row.keys())
            if not file_exists:
                writer.writeheader()
            writer.writerow(row)
            f.flush()
            os.fsync(f.fileno())

        return float(metrics["pnl"])

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=trials)
