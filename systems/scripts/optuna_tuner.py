from __future__ import annotations

"""Optuna-based tuning of strategy knobs via live engine."""

import copy
from typing import Dict

import optuna

from systems.live_engine import run_live
from systems.scripts.ledger import Ledger
from systems.utils.addlog import addlog
from systems.utils.path import find_project_root
from systems.utils.settings_loader import load_settings


def run_tuner(tag: str, verbose: int = 0, trials: int = 20) -> None:
    """Run Optuna tuner for a specific ``tag``."""

    base_settings = load_settings()
    root = find_project_root()
    ledger_path = root / "data" / "ledgers" / f"{tag}.json"

    def objective(trial: optuna.trial.Trial) -> float:
        settings = copy.deepcopy(base_settings)
        knobs: Dict[str, float | int] = {
            "buy_cooldown": trial.suggest_int("buy_cooldown", 1, 24),
            "sell_cooldown": trial.suggest_int("sell_cooldown", 4, 24),
            "min_roi": trial.suggest_float("min_roi", 0.1, 1.0),
            "buy_floor": trial.suggest_float("buy_floor", 0.1, 0.5),
            "sell_ceiling": trial.suggest_float("sell_ceiling", 0.5, 1.0),
        }
        settings["knobs"] = knobs

        if ledger_path.exists():
            backup = ledger_path.read_text(encoding="utf-8")
        else:
            backup = None

        run_live(tag=tag, dry=True, verbose=verbose, settings_override=settings)

        ledger = Ledger.load_ledger(tag)
        closed = ledger.get_closed_notes()
        realized_gain = sum(n.get("gain", 0.0) for n in closed)
        capital_used = sum(
            n.get("entry_price", 0.0) * n.get("entry_amount", 0.0) for n in closed
        )
        capital_returned = sum(
            n.get("exit_price", 0.0) * n.get("entry_amount", 0.0) for n in closed
        )
        penalty = 0.001
        score = realized_gain - (capital_used + capital_returned) * penalty

        if backup is None:
            ledger_path.unlink(missing_ok=True)
        else:
            ledger_path.write_text(backup, encoding="utf-8")

        return score

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=trials, show_progress_bar=verbose >= 1)

    best = study.best_trial
    addlog(
        f"[TUNE] best score {best.value:.4f} params {best.params}",
        verbose_int=1,
        verbose_state=verbose,
    )
