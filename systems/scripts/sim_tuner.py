from __future__ import annotations

"""Single-window simulation tuner using Optuna."""

import copy
import csv
import json
import os
import re
from pathlib import Path
from typing import Any, Dict

import optuna

from systems.sim_engine import run_simulation
from systems.scripts.fetch_canles import fetch_candles
from systems.scripts.ledger import Ledger
from systems.utils.addlog import addlog
from systems.utils.path import find_project_root

def _load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)

def run_sim_tuner(tag: str, window: str, verbose: int = 0) -> None:
    """Run Optuna tuning on a single window for ``tag`` and ``window``."""
    tag = tag.upper()
    window = window.lower()
    root = find_project_root()
    settings_path = root / "settings" / "settings.json"
    knobs_path = root / "settings" / "knobs.json"

    base_settings = _load_json(settings_path)
    knobs_cfg = _load_json(knobs_path).get(tag)
    if knobs_cfg is None:
        raise ValueError(f"No knob configuration found for tag: {tag}")

    window_knobs = knobs_cfg.get(window)
    if not window_knobs:
        raise ValueError(f"No knob ranges found for window: {window}")

    ledger_key = None
    for name, cfg in base_settings.get("ledger_settings", {}).items():
        if cfg.get("tag", "").upper() == tag:
            ledger_key = name
            break
    if ledger_key is None:
        raise ValueError(f"Tag {tag} not present in settings")

    init_capital = float(base_settings.get("simulation_capital", 0))

    results_path = root / "data" / "tmp" / "sim_tune_results.csv"
    results_path.parent.mkdir(parents=True, exist_ok=True)
    trials_dir = root / "data" / "tmp" / "tune_trials"
    trials_dir.mkdir(parents=True, exist_ok=True)

    safe_tag = re.sub(r"[^a-z0-9_]+", "_", tag.lower())
    safe_window = re.sub(r"[^a-z0-9_]+", "_", window.lower())
    trial_csv_path = trials_dir / f"{safe_tag}_{safe_window}.csv"
    trial_fieldnames = ["trial_number", "score", *window_knobs.keys()]

    import systems.utils.settings_loader as settings_loader
    import systems.sim_engine as sim_engine

    def objective(trial: optuna.trial.Trial) -> float:
        trial_settings = copy.deepcopy(base_settings)
        w_settings = trial_settings["ledger_settings"][ledger_key]["window_settings"]
        current_cfg = w_settings[window]

        for knob, bounds in window_knobs.items():
            if isinstance(bounds, dict):
                low = bounds.get("low") or bounds.get("min")
                high = bounds.get("high") or bounds.get("max")
            else:
                low, high = bounds
            base_val = current_cfg.get(knob)
            if (
                isinstance(base_val, int)
                and isinstance(low, int)
                and isinstance(high, int)
            ):
                value = trial.suggest_int(knob, int(low), int(high))
            else:
                value = trial.suggest_float(knob, float(low), float(high))
            current_cfg[knob] = value

        original_loader = settings_loader.load_settings
        original_sim_loader = sim_engine.load_settings
        settings_loader.load_settings = lambda: trial_settings
        sim_engine.load_settings = lambda: trial_settings
        try:
            run_simulation(tag, verbose)
        finally:
            settings_loader.load_settings = original_loader
            sim_engine.load_settings = original_sim_loader

        ledger = Ledger.load_ledger(tag, sim=True)
        final_price = float(fetch_candles(tag).iloc[-1]["close"])
        summary = ledger.get_account_summary(final_price)
        open_value = summary.get("open_value", 0.0)
        realized_gain = summary.get("realized_gain", 0.0)
        open_cost = sum(
            n.get("entry_price", 0.0) * n.get("entry_amount", 0.0)
            for n in ledger.get_open_notes()
        )
        idle_capital = init_capital + realized_gain - open_cost
        penalty = 0.01
        score = realized_gain - penalty * (idle_capital + open_value)
        if verbose:
            addlog(
                f"[TUNE][{window}] Trial {trial.number} score={score:.4f}",
                verbose_int=1,
                verbose_state=verbose,
            )
        row = {
            "trial_number": trial.number,
            "score": score,
            **{k: trial.params.get(k) for k in window_knobs.keys()},
        }
        file_exists = trial_csv_path.exists()
        with trial_csv_path.open("a", newline="", encoding="utf-8") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=trial_fieldnames)
            if not file_exists:
                writer.writeheader()
            writer.writerow(row)
            csvfile.flush()
            os.fsync(csvfile.fileno())
        return score

    if verbose <= 0:
        optuna.logging.set_verbosity(optuna.logging.WARNING)
    else:
        optuna.logging.set_verbosity(optuna.logging.INFO)

    study = optuna.create_study(direction="maximize")
    print("[TUNER] Running indefinitely. Press CTRL+C to stop.")
    try:
        study.optimize(objective, n_trials=None)
    except KeyboardInterrupt:
        print("[TUNER] Interrupted by user. Saving progress...")

    best_score = study.best_value
    best_params = study.best_params

    addlog(
        f"[TUNE][{window}] Best parameters: {best_params}",
        verbose_int=1,
        verbose_state=verbose,
    )

    row = {
        "tag": tag,
        "window": window,
        "score": best_score,
        **best_params,
    }
    file_exists = results_path.exists()
    with results_path.open("a", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=row.keys())
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)
        csvfile.flush()
        os.fsync(csvfile.fileno())
