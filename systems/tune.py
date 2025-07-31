from __future__ import annotations

"""Optuna-based hyperparameter tuner for window strategies."""

import json
from typing import Any, Dict

import optuna

from systems.utils.logger import addlog
from systems.utils.path import find_project_root
from systems.sim_engine import run_simulation


_DEF_ROLE = "fish"


def _load_knobs() -> Dict[str, Dict[str, Any]]:
    """Load knob definitions from ``settings/knobs.json``."""
    root = find_project_root()
    path = root / "settings" / "knobs.json"
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def run_tuner(
    *,
    tag: str,
    window: str,
    role: str = _DEF_ROLE,
    trials: int = 20,
    verbose: int = 0,
) -> None:
    """Run Optuna tuner for a single role."""
    tag = tag.upper()
    knobs = _load_knobs().get(role)
    if knobs is None:
        raise ValueError(f"Unknown role '{role}' in knobs.json")

    root = find_project_root()
    settings_path = root / "settings" / "settings.json"
    with settings_path.open("r", encoding="utf-8") as f:
        base_settings = json.load(f)

    def objective(trial: optuna.trial.Trial) -> float:
        settings = json.loads(json.dumps(base_settings))
        window_cfg = settings.setdefault("general_settings", {}).setdefault("windows", {}).setdefault(role, {})
        for key, spec in knobs.items():
            if isinstance(spec, list) and spec and isinstance(spec[0], str):
                val = trial.suggest_categorical(key, spec)
            elif isinstance(spec, list) and len(spec) == 2:
                low, high = spec
                if isinstance(low, int) and isinstance(high, int):
                    val = trial.suggest_int(key, low, high)
                else:
                    val = trial.suggest_float(key, float(low), float(high))
            else:
                raise ValueError(f"Invalid spec for {key}: {spec}")
            window_cfg[key] = val

        import systems.sim_engine as sim_engine  # local import for patching

        def _patched_load_settings():
            return settings

        sim_engine.load_settings = _patched_load_settings

        run_simulation(tag=tag, verbose=0)

        ledger_path = root / "data" / "tmp" / "ledgersimulation.json"
        with ledger_path.open("r", encoding="utf-8") as f:
            ledger = json.load(f)

        pnl = float(ledger.get("pnl", 0.0))
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
        if capital_used == 0:
            return 0.0
        return pnl / capital_used

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=trials)

    best = study.best_params
    addlog(f"[TUNE] Best params for {role}: {best}", verbose_int=1, verbose_state=verbose)

    out_dir = root / "data" / "tmp"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"tune_best_{role}.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(best, f, indent=2)

    print(f"Best parameters for {role}: {best}")

