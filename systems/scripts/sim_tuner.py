from __future__ import annotations

"""Sequential per-window simulation tuner using Optuna."""

import copy
import csv
import json
import os
from collections import OrderedDict
from typing import Any, Dict

import optuna

from systems.sim_engine import run_simulation
from systems.scripts.fetch_candles import load_coin_csv
from systems.scripts.ledger import load_ledger
from systems.utils.addlog import addlog
from systems.utils.config import (
    load_ledger_config,
    load_settings,
    resolve_path,
)
from systems.utils.resolve_symbol import split_tag, resolve_symbols, to_tag
import ccxt


def run_sim_tuner(*, ledger: str, verbose: int = 0) -> None:
    """Run sequential Optuna tuning on each window for ``ledger``."""

    ledger_cfg = load_ledger_config(ledger)
    client = ccxt.kraken({"enableRateLimit": True})
    symbols = resolve_symbols(client, ledger_cfg["kraken_name"])
    tag = to_tag(symbols["kraken_name"]).upper()
    window_settings = ledger_cfg.get("window_settings", {})
    if not window_settings:
        raise ValueError("No windows defined for ledger")

    root = resolve_path("")
    settings = load_settings(reload=True)
    knobs_path = root / "settings" / "knobs.json"
    with knobs_path.open("r", encoding="utf-8") as f:
        knobs_cfg = json.load(f).get(tag)
    if knobs_cfg is None:
        raise ValueError(f"No knob configuration found for tag: {tag}")

    init_capital = float(settings.get("simulation_capital", 0))
    results_path = root / "data" / "tmp" / "simtune_results.csv"
    results_path.parent.mkdir(parents=True, exist_ok=True)
    best_knobs: Dict[str, Any] = {}

    import systems.utils.config as config_mod
    import systems.sim_engine as sim_engine

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
            trial_settings = copy.deepcopy(settings)
            trial_ledger_cfg = copy.deepcopy(ledger_cfg)

            # Freeze previously tuned windows
            w_settings = trial_ledger_cfg.get("window_settings", {})
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

            trial_settings["ledger_settings"][ledger] = trial_ledger_cfg

            original_load_settings = config_mod.load_settings
            original_load_ledger = config_mod.load_ledger_config
            config_mod.load_settings = lambda reload=False: trial_settings
            config_mod.load_ledger_config = (
                lambda name: trial_settings["ledger_settings"][name]
            )

            original_sim_loader = (
                sim_engine.load_settings if hasattr(sim_engine, "load_settings") else None
            )
            if original_sim_loader:
                sim_engine.load_settings = lambda reload=False: trial_settings

            try:
                run_simulation(ledger=ledger, verbose=verbose, window_names=[window_name])
            finally:
                config_mod.load_settings = original_load_settings
                config_mod.load_ledger_config = original_load_ledger
                if original_sim_loader:
                    sim_engine.load_settings = original_sim_loader

            ledger_obj = load_ledger(ledger, tag=tag, sim=True)
            base, _ = split_tag(tag)
            final_price = float(load_coin_csv(base).iloc[-1]["close"])
            summary = ledger_obj.get_account_summary(final_price)
            open_value = summary.get("open_value", 0.0)
            realized_gain = summary.get("realized_gain", 0.0)
            open_cost = sum(
                n.get("entry_price", 0.0) * n.get("entry_amount", 0.0)
                for n in ledger_obj.get_open_notes()
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
        interrupted = False

        try:
            while True:
                study.optimize(objective, n_trials=1)
                trial = study.trials[-1]
                row = OrderedDict(
                    [
                        ("ledger", ledger),
                        ("window", window_name),
                        ("trial", trial.number),
                        ("score", trial.value),
                    ]
                )
                for k, v in trial.params.items():
                    row[k] = v
                file_exists = results_path.exists()
                with results_path.open("a", newline="", encoding="utf-8") as csvfile:
                    writer = csv.DictWriter(csvfile, fieldnames=row.keys())
                    if not file_exists:
                        writer.writeheader()
                    writer.writerow(row)
                    csvfile.flush()
                    os.fsync(csvfile.fileno())
        except KeyboardInterrupt:
            addlog(
                "[TUNE] Interrupted by user, shutting down.",
                verbose_int=1,
                verbose_state=verbose,
            )
            interrupted = True

        best_score = study.best_value
        best_params = study.best_params
        best_knobs[window_name] = best_params

        addlog(
            f"[TUNE][{window_name}] Best parameters: {best_params}",
            verbose_int=1,
            verbose_state=verbose,
        )

        if interrupted:
            return

