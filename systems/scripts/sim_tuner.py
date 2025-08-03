from __future__ import annotations

"""Sequential per-window simulation tuner using Optuna."""

import copy
import json
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


def run_sim_tuner(tag: str, verbose: int = 0) -> None:
    """Run sequential Optuna tuning on each window for ``tag``."""
    tag = tag.upper()
    root = find_project_root()
    settings_path = root / "settings" / "settings.json"
    knobs_path = root / "settings" / "knobs.json"

    base_settings = _load_json(settings_path)
    knobs_cfg = _load_json(knobs_path).get(tag)
    if knobs_cfg is None:
        raise ValueError(f"No knob configuration found for tag: {tag}")

    ledger_key = None
    for name, cfg in base_settings.get("ledger_settings", {}).items():
        if cfg.get("tag", "").upper() == tag:
            ledger_key = name
            break
    if ledger_key is None:
        raise ValueError(f"Tag {tag} not present in settings")

    init_capital = float(base_settings.get("simulation_capital", 0))

    out_dir = root / "data" / "tmp" / "best_knobs"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{tag}.json"
    if out_path.exists():
        with out_path.open("r", encoding="utf-8") as f:
            best_knobs: Dict[str, Any] = json.load(f)
    else:
        best_knobs = {}

    import systems.utils.settings_loader as settings_loader
    import systems.sim_engine as sim_engine

    window_settings = base_settings["ledger_settings"][ledger_key]["window_settings"]
    for window_name in window_settings:
        window_knobs = knobs_cfg.get(window_name)
        if not window_knobs:
            if verbose:
                addlog(
                    f"[TUNE] No knob ranges for window '{window_name}', skipping",
                    verbose_int=1,
                    verbose_state=verbose,
                )
            continue

        def objective(trial: optuna.trial.Trial) -> float:
            trial_settings = copy.deepcopy(base_settings)
            w_settings = trial_settings["ledger_settings"][ledger_key]["window_settings"]

            # Freeze previously tuned windows
            for w, params in best_knobs.items():
                if w in w_settings:
                    w_settings[w].update(params)

            current_cfg = w_settings[window_name]
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

            # Inject trial settings
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
                    f"[TUNE][{window_name}] Trial {trial.number} score={score:.4f}",
                    verbose_int=1,
                    verbose_state=verbose,
                )
            return score

        if verbose <= 0:
            optuna.logging.set_verbosity(optuna.logging.WARNING)
        else:
            optuna.logging.set_verbosity(optuna.logging.INFO)

        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=50)

        addlog(
            f"[TUNE][{window_name}] Best parameters: {study.best_params}",
            verbose_int=1,
            verbose_state=verbose,
        )

        best_knobs[window_name] = study.best_params
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(best_knobs, f, indent=2)


