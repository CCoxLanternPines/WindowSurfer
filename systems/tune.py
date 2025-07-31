from __future__ import annotations

"""Optuna-based hyperparameter tuner for multi-window strategies.

This module was inspired by the single-window tuner found in ``/legacy``.
It expands the approach so that each *window role* (``fish``, ``whale``,
``knife``...) can be tuned simultaneously.  Parameters for every role are
read from :mod:`settings.knobs`, sampled with Optuna and then injected into
the simulation settings as a composite configuration.
"""

import json
from pathlib import Path
from typing import Any, Dict, List

import optuna
import pandas as pd

from systems.utils.logger import addlog
from systems.utils.path import find_project_root
from systems.sim_engine import run_simulation


def _load_knobs() -> Dict[str, Dict[str, Any]]:
    """Load knob definitions from ``settings/knobs.json``."""
    root = find_project_root()
    path = root / "settings" / "knobs.json"
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def run_tuner(*, tag: str, trials: int = 20, verbose: int = 0) -> None:
    """Run Optuna tuner across all defined window roles."""
    tag = tag.upper()
    knob_space = _load_knobs()

    root = find_project_root()
    settings_path = root / "settings" / "settings.json"
    with settings_path.open("r", encoding="utf-8") as f:
        base_settings = json.load(f)

    trial_records: List[Dict[str, Any]] = []

    def objective(trial: optuna.trial.Trial) -> float:
        print(f"[TUNE] Trial {trial.number} started")
        settings = json.loads(json.dumps(base_settings))
        trial_windows: Dict[str, Dict[str, Any]] = {}
        flat_params: Dict[str, Any] = {}
        for role, knobs in knob_space.items():
            role_cfg: Dict[str, Any] = {}
            for key, spec in knobs.items():
                name = f"{role}_{key}"
                if isinstance(spec, list) and spec and isinstance(spec[0], str):
                    val = trial.suggest_categorical(name, spec)
                elif isinstance(spec, list) and len(spec) == 2:
                    low, high = spec
                    if isinstance(low, int) and isinstance(high, int):
                        val = trial.suggest_int(name, low, high)
                    else:
                        val = trial.suggest_float(name, float(low), float(high))
                else:
                    raise ValueError(f"Invalid spec for {key}: {spec}")
                role_cfg[key] = val
                flat_params[name] = val
            trial_windows[role] = role_cfg
        settings.setdefault("general_settings", {})["windows"] = trial_windows

        import systems.sim_engine as sim_engine  # local import for patching

        def _patched_load_settings():
            return settings

        sim_engine.load_settings = _patched_load_settings

        run_simulation(tag=tag, verbose=0)

        print("[TUNE] Simulation completed, reading ledger...")
        ledger_path = root / "data" / "tmp" / "ledgersimulation.json"
        with ledger_path.open("r", encoding="utf-8") as f:
            ledger = json.load(f)

        net_pnl = float(ledger.get("pnl", 0.0))
        notes = ledger.get("open_notes", []) + ledger.get("closed_notes", [])
        events = []
        for note in notes:
            entry = int(note.get("entry_tick", 0))
            exit_ = note.get("exit_tick")
            exit_tick = int(exit_) if exit_ is not None else entry + 10**9
            amt = float(note.get("entry_usdt", 0.0))
            events.append((entry, amt))
            events.append((exit_tick, -amt))
        active = 0.0
        capital_used = 0.0
        for tick, delta in sorted(events, key=lambda x: x[0]):
            active += delta
            capital_used = max(capital_used, active)
        score = 0.0 if capital_used == 0 else net_pnl / capital_used
        trial_records.append(
            {
                "score": score,
                "net_gain": net_pnl,
                "capital_used": capital_used,
                **flat_params,
            }
        )
        return score

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=trials)

    best_flat = study.best_params
    addlog(
        f"[TUNE] Best params: {best_flat}",
        verbose_int=1,
        verbose_state=verbose,
    )

    best_nested: Dict[str, Dict[str, Any]] = {}
    for role, knobs in knob_space.items():
        best_nested[role] = {
            key: best_flat[f"{role}_{key}"] for key in knobs.keys()
        }

    out_dir = Path(root / "data" / "tmp")
    out_dir.mkdir(parents=True, exist_ok=True)

    leaderboard_path = out_dir / "tune_leaderboard.csv"
    df = pd.DataFrame(trial_records)
    df.sort_values("score", ascending=False, inplace=True)
    df.to_csv(leaderboard_path, index=False)
    if verbose >= 1:
        print("\n🏆 Top 10 Tuning Results:")
        top = df.head(10)
        ordered = [
            "score",
            "net_gain",
            "capital_used",
            *[c for c in top.columns if c not in {"score", "net_gain", "capital_used"}],
        ]
        print(top[ordered].to_string(index=False))

    out_path = out_dir / "tune_best_multi.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(best_nested, f, indent=2)

    print(f"Best parameters: {best_nested}")

