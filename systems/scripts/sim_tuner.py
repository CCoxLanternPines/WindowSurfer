from __future__ import annotations

"""Simulation-based parameter tuner using Optuna."""

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
    """Tune simulation knobs for ``tag`` using Optuna."""
    tag = tag.upper()
    root = find_project_root()
    settings_path = root / "settings" / "settings.json"
    knobs_path = root / "settings" / "knobs.json"

    base_settings = _load_json(settings_path)
    knob_cfg = _load_json(knobs_path).get(tag)
    if knob_cfg is None:
        raise ValueError(f"No knob configuration found for tag: {tag}")

    ledger_key = None
    for name, cfg in base_settings.get("ledger_settings", {}).items():
        if cfg.get("tag", "").upper() == tag:
            ledger_key = name
            break
    if ledger_key is None:
        raise ValueError(f"Tag {tag} not present in settings")

    init_capital = float(base_settings.get("simulation_capital", 0))

    import systems.utils.settings_loader as settings_loader
    import systems.sim_engine as sim_engine

    def objective(trial: optuna.trial.Trial) -> float:
        trial_settings = copy.deepcopy(base_settings)
        fish_cfg = (
            trial_settings["ledger_settings"][ledger_key]["window_settings"]["fish"]
        )

        for knob, bounds in knob_cfg.items():
            if isinstance(bounds, dict):
                low = bounds.get("low") or bounds.get("min")
                high = bounds.get("high") or bounds.get("max")
            else:
                low, high = bounds
            base_val = fish_cfg.get(knob)
            if isinstance(base_val, int) and isinstance(low, int) and isinstance(high, int):
                value = trial.suggest_int(knob, int(low), int(high))
            else:
                value = trial.suggest_float(knob, float(low), float(high))
            fish_cfg[knob] = value

        # Monkeypatch settings loader to inject trial settings
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
                f"[TUNE] Trial {trial.number} score={score:.4f}",
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
        f"[TUNE] Best parameters: {study.best_params}",
        verbose_int=1,
        verbose_state=verbose,
    )

    out_dir = root / "data" / "tmp" / "best_knobs"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{tag}.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(study.best_params, f, indent=2)

