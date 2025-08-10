from __future__ import annotations

import json
from pathlib import Path
from typing import List, Dict

import numpy as np
import optuna
import pandas as pd

from systems.utils.imports import ensure_project_root

ensure_project_root()

from systems.sim_engine import RUNNER_ID, run_sim_blocks


def _bootstrap_centroids_from_brain(tag: str, run_id: str, cent_path: Path) -> None:
    """Create centroids file from the latest brain if missing."""
    brain_dir = Path(f"data/regimes/{run_id}/brains")
    brain_path: Path | None = None
    if brain_dir.exists():
        direct = brain_dir / f"brain_{tag}.json"
        if direct.exists():
            brain_path = direct
        else:
            candidates = sorted(brain_dir.glob(f"brain_{tag}__k*_seed*_*"))
            if candidates:
                brain_path = max(candidates, key=lambda p: p.stat().st_mtime)
    if brain_path is None:
        raise SystemExit(
            f"[TUNE] No centroids or brain found for {tag} under run-id {run_id}. Run the cluster step first."
        )

    with brain_path.open() as fh:
        brain = json.load(fh)
    centroids = None
    for key in ["centroids", "kmeans_centroids", "centers", "cluster_centers_"]:
        if key in brain:
            centroids = brain[key]
            break
    if centroids is None:
        raise SystemExit(
            f"[TUNE] No centroids or brain found for {tag} under run-id {run_id}. Run the cluster step first."
        )

    k = brain.get("k") or len(centroids)
    seed = brain.get("seed")
    feature_names = brain.get("feature_names") or brain.get("features") or []

    scaler = brain.get("scaler", {})
    payload = {
        "k": k,
        "seed": seed,
        "feature_names": feature_names,
        "centroids": centroids,
    }
    if feature_names:
        payload["features"] = feature_names
    if scaler.get("mean") is not None and scaler.get("std") is not None:
        payload.update(
            {
                "mean": scaler.get("mean"),
                "std": scaler.get("std"),
                "std_floor": scaler.get("std_floor", 1e-6),
            }
        )
    if brain.get("feature_sha"):
        payload["feature_sha"] = brain.get("feature_sha")

    cent_path.parent.mkdir(parents=True, exist_ok=True)
    with cent_path.open("w") as fh:
        json.dump(payload, fh, indent=2)
    print(f"[TUNE] Bootstrapped centroids from brain → {cent_path}")
    print(
        f"[AUDIT] centroids:k={k} seed={seed} features={len(feature_names)} path={cent_path} ok"
    )


def run_regime_tuning(
    tag: str,
    run_id: str,
    regime_id: int,
    tau: float,
    trials: int,
    metric: str = "pnl_dd",
    seed: int = 2,
    verbose: int = 0,
    smoke: bool = False,
    write_seed: bool = False,
) -> None:
    """Run Optuna tuning for a specific regime over pure blocks.

    Parameters
    ----------
    tag: str
        Asset tag.
    run_id: str
        Identifier for artifacts.
    regime_id: int
        Regime to tune.
    tau: float
        Purity threshold.
    trials: int
        Number of Optuna trials.
    metric: str
        Objective metric (pnl_dd, pnl, sharpe_like).
    seed: int
        Random seed.
    verbose: int
        Verbosity level.
    smoke: bool
        If True, perform a lightweight discovery run and exit.
    write_seed: bool
        Update regimes/seed_knobs.json with best params.
    """

    # Lazy imports to avoid hard dependencies when module is imported
    from .data_loader import load_or_fetch
    from .block_planner import plan_blocks, parse_duration
    from .features import extract_all_features
    from .regime_cluster import align_centroids
    from .purity import compute_purity

    print(f"[TUNE] Using sim runner: {RUNNER_ID}")

    # ------------------------------------------------------------------
    # Load settings for block planning
    # ------------------------------------------------------------------
    settings_path = Path("settings/settings.json")
    if not settings_path.exists():
        settings_path = Path("settings.json")
    settings: Dict[str, Dict] = {}
    if settings_path.exists():
        with settings_path.open() as fh:
            settings = json.load(fh)
    rset = settings.get("regime_settings", {})
    train_cfg = rset.get("train", "3w")
    test_cfg = rset.get("test", "1m")
    step_cfg = rset.get("step", "1m")

    # ------------------------------------------------------------------
    # Load data and compute blocks/features
    # ------------------------------------------------------------------
    df = load_or_fetch(tag)
    train_len = parse_duration(train_cfg)
    test_len = parse_duration(test_cfg)
    step_len = parse_duration(step_cfg)
    blocks = plan_blocks(df, train_len, test_len, step_len)

    feats_df = extract_all_features(df, blocks)

    # ------------------------------------------------------------------
    # Load centroids and assign regimes
    # ------------------------------------------------------------------
    cent_path = Path(f"data/regimes/{run_id}/centroids/centroids_{tag}.json")
    if not cent_path.exists():
        _bootstrap_centroids_from_brain(tag, run_id, cent_path)
    with cent_path.open() as fh:
        centroids = json.load(fh)
    if "feature_names" in centroids and "features" not in centroids:
        centroids["features"] = centroids["feature_names"]
    centroids = align_centroids(
        {
            "features": centroids["features"],
            "feature_sha": centroids.get("feature_sha"),
            "mean": centroids["mean"],
            "std": centroids["std"],
            "std_floor": centroids.get("std_floor", 1e-6),
        },
        centroids,
    )
    feature_names = centroids["features"]
    mean = np.array(centroids["mean"], dtype=float)
    std = np.maximum(np.array(centroids["std"], dtype=float), centroids.get("std_floor", 1e-6))
    X = feats_df[feature_names].to_numpy()
    Z = (X - mean) / std
    C = np.array(centroids["centroids"], dtype=float)
    d = ((Z[:, None, :] - C[None, :, :]) ** 2).sum(axis=2)
    labels = d.argmin(axis=1)
    feats_df["regime_id"] = labels
    feats_df["block_id"] = feats_df["block_id"].astype(int)

    # ------------------------------------------------------------------
    # Purity filtering
    # ------------------------------------------------------------------
    try:
        purity_path = compute_purity(tag=tag, run_id=run_id, tau=tau, win_dur="1w", stride=6)
        purity_df = pd.read_csv(purity_path)
        purity_col = f"purity{regime_id}"
        if purity_col not in purity_df.columns:
            purity_df = pd.DataFrame(
                {"block_id": feats_df["block_id"], purity_col: (labels == regime_id).astype(float)}
            )
    except Exception:
        purity_col = f"purity{regime_id}"
        purity_df = pd.DataFrame(
            {"block_id": feats_df["block_id"], purity_col: (labels == regime_id).astype(float)}
        )

    pure_blocks = purity_df[purity_df[purity_col] >= tau]["block_id"].astype(int).tolist()
    if not pure_blocks:
        print(f"[TUNE] No pure blocks for R{regime_id} at τ={tau}")
        raise SystemExit(1)

    ranges = []
    for b_id in pure_blocks:
        block = blocks[b_id - 1]  # block_id is 1-indexed
        ranges.append((block["test_start"], block["test_end"]))

    print(f"[TUNE] Ranges: {len(ranges)} blocks | τ={tau:.2f} | regime=R{regime_id}")
    if smoke:
        return

    # ------------------------------------------------------------------
    # Optuna search space
    # ------------------------------------------------------------------
    leaderboard: List[Dict[str, float]] = []

    def objective(trial: optuna.trial.Trial) -> float:
        knobs = {
            "position_pct": trial.suggest_float("position_pct", 0.02, 0.12),
            "max_concurrent": trial.suggest_int("max_concurrent", 1, 3),
            "buy_cooldown": trial.suggest_int("buy_cooldown", 4, 18),
            "volatility_gate": trial.suggest_float("volatility_gate", 0.6, 1.2),
            "rsi_buy": trial.suggest_int("rsi_buy", 30, 45),
            "take_profit": trial.suggest_float("take_profit", 0.008, 0.035),
            "trailing_stop": trial.suggest_float("trailing_stop", 0.006, 0.03),
            "stop_loss": trial.suggest_float("stop_loss", 0.02, 0.08),
            "sell_cooldown": trial.suggest_int("sell_cooldown", 3, 16),
        }
        result = run_sim_blocks(
            tag=tag,
            ranges=ranges,
            knobs=knobs,
            verbose=verbose >= 2,
        )
        pnl = float(result.get("pnl", 0.0))
        maxdd = float(result.get("maxdd", 0.0))
        trades = int(result.get("trades", 0))
        returns = np.asarray(result.get("returns", []), dtype=float)
        if trades < 10:
            objective_val = -1e12
        else:
            pnl_dd = pnl * (1 - 1.5 * maxdd)
            if metric == "pnl":
                objective_val = pnl
            elif metric == "sharpe_like" and returns.size > 1 and returns.std() > 0:
                objective_val = float(returns.mean() / returns.std())
            else:
                objective_val = pnl_dd
        leaderboard.append(
            {
                "trial": trial.number,
                "pnl": pnl,
                "maxdd": maxdd,
                "pnl_dd": pnl * (1 - 1.5 * maxdd),
                "trades": trades,
                **knobs,
            }
        )
        if verbose:
            print(
                f"[TUNE] trial={trial.number} pnl={pnl:.2f} maxdd={maxdd:.3f} trades={trades} obj={metric}={objective_val:.2f}"
            )
            if verbose >= 3:
                tb = result.get("trades_by_block")
                if isinstance(tb, list):
                    block_map = {pure_blocks[i]: tb[i] for i in range(min(len(tb), len(pure_blocks)))}
                    print(f"[TUNE] block_trades={block_map}")
        return objective_val

    sampler = optuna.samplers.TPESampler(seed=seed)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    study.optimize(objective, n_trials=trials, show_progress_bar=False)

    lb_df = pd.DataFrame(leaderboard)
    out_dir = Path(f"data/regimes/{run_id}/tuning/{tag}/R{regime_id}")
    out_dir.mkdir(parents=True, exist_ok=True)
    leader_path = out_dir / "leaderboard.csv"
    lb_df.to_csv(leader_path, index=False)

    best_params = study.best_trial.params
    best_row = lb_df.loc[lb_df["trial"] == study.best_trial.number].iloc[0]
    best_path = out_dir / "best.json"
    with best_path.open("w") as fh:
        json.dump(best_params, fh, indent=2)

    print(
        f"[TUNE] BestTrial R{regime_id} τ={tau} | pnl={best_row['pnl']:.2f} maxdd={best_row['maxdd']:.3f} pnl_dd={best_row['pnl_dd']:.2f} trades={int(best_row['trades'])}"
    )
    param_str = ", ".join(f"{k}={v}" for k, v in best_params.items())
    print(f"[TUNE] Params: {param_str}")
    print(f"[TUNE] Saved: {leader_path} | {best_path}")

    if write_seed:
        seed_path = Path("regimes/seed_knobs.json")
        seed_path.parent.mkdir(parents=True, exist_ok=True)
        data = {}
        if seed_path.exists():
            with seed_path.open() as fh:
                data = json.load(fh)
        regime_key = f"R{regime_id}"
        regime_data = data.setdefault(regime_key, {})
        regime_data[tag] = best_params
        with seed_path.open("w") as fh:
            json.dump(data, fh, indent=2)
