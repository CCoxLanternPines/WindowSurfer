from __future__ import annotations

import copy
import csv
import json
import os
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict

import optuna

from sim_engine import run_sim


def unflatten_dict(flat: dict[str, float]) -> dict:
    nested: dict[str, Any] = {}
    for k, v in flat.items():
        parts = k.split(".")
        d = nested
        for p in parts[:-1]:
            d = d.setdefault(p, {})
        d[parts[-1]] = v
        
    return nested


def run(trials: int) -> None:
    """Run an Optuna study for the provided number of ``trials``."""
    with open("settings.json", "r", encoding="utf-8") as f:
        base_cfg: Dict[str, Any] = json.load(f)
    with open("knobs.json", "r", encoding="utf-8") as f:
        knob_ranges: "OrderedDict[str, list[float]]" = json.load(
            f, object_pairs_hook=OrderedDict
        )

    results_path = Path("tune_results.csv")
    knob_keys = list(knob_ranges.keys())

    def objective(trial: optuna.trial.Trial) -> float:
        params: Dict[str, float] = {}
        for name, bounds in knob_ranges.items():
            low, high = bounds
            if isinstance(low, int) and isinstance(high, int):
                value = trial.suggest_int(name, int(low), int(high))
            else:
                value = trial.suggest_float(name, float(low), float(high))
            params[name] = value

        cfg = copy.deepcopy(base_cfg)
        nested_params = unflatten_dict(params)
        for key, sub in nested_params.items():
            if isinstance(sub, dict) and key in cfg and isinstance(cfg[key], dict):
                cfg[key].update(sub)
            else:
                cfg[key] = sub

        metrics = run_sim(cfg)

        row = OrderedDict([("trial", trial.number)])
        for k in knob_keys:
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
                        for line in existing_content.splitlines()[1:]:
                            f.write(line + "\n")
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
