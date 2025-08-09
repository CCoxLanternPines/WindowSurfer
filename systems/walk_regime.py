from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Callable, Dict, Any

import numpy as np

from . import data_loader, features, regime_cluster, policy_blender, sim_engine, optimizer

SETTINGS_PATH = Path(__file__).resolve().parent.parent / 'settings.json'
RESULTS_PATH = Path('regime_walk_results.csv')


def _load_settings() -> dict:
    with SETTINGS_PATH.open() as fh:
        return json.load(fh)


def run(
    *,
    tag: str,
    train: str,
    test: str,
    step: str,
    clusters: int,
    microtrials: int,
    fees: float,
    slip: float,
    hysteresis: int,
    blend: str,
) -> None:
    settings = _load_settings()
    prices = data_loader.load_prices(tag)
    train_len = data_loader.parse_window(train)
    test_len = data_loader.parse_window(test)
    step_len = data_loader.parse_window(step)
    feat_win = data_loader.parse_window(
        settings.get('regime_settings', {}).get('feature_window', '30d')
    )
    hyst = policy_blender.HysteresisRegime(hysteresis)
    blend_alpha = settings.get('regime_settings', {}).get('blend_alpha', 1.0)
    sim_cap = settings.get('simulation_capital', 1000)
    seeds_mode = 'blend' if blend != 'none' else 'seeds'

    rows = []
    cursor = 0
    while cursor + train_len + test_len <= len(prices):
        train_slice = data_loader.slice_prices(prices, cursor, cursor + train_len)
        X_train = features.compute_window_features(train_slice, feat_win)
        model = regime_cluster.fit_kmeans(X_train, clusters)
        regime_cluster.save_centroids(tag, clusters, model)

        block_start = cursor + train_len
        assign_window = data_loader.slice_prices(prices, block_start - feat_win, block_start)
        X_block = features.compute_window_features(assign_window, feat_win)
        x_feat = X_block[-1]
        idx, dists = regime_cluster.assign_regime(x_feat, model)
        regime_id = f"R{idx}"
        effective = hyst.update(regime_id)
        policy = policy_blender.blend_policy(dists, blend, blend_alpha)
        policy_source = seeds_mode
        if microtrials > 0:
            policy = optimizer.run(
                trials=microtrials,
                prices=data_loader.slice_prices(prices, block_start - test_len, block_start),
                base_policy=policy,
            )
            policy_source = 'micro'

        def provider(_: int) -> Dict[str, Any]:
            return policy

        metrics = sim_engine.run_sim(
            prices=prices,
            base_settings={'capital': sim_cap},
            policy_provider=provider,
            start_idx=block_start,
            end_idx=block_start + test_len,
            fees_bps=fees,
            slip_bps=slip,
        )
        rows.append(
            {
                'start_idx': block_start,
                'end_idx': block_start + test_len,
                'regime_id': effective,
                'policy_source': policy_source,
                'pnl': metrics.get('pnl', 0.0),
                'max_dd': metrics.get('max_dd', 0.0),
                'trades': metrics.get('trades', 0),
                'avg_hold': metrics.get('avg_hold', 0.0),
                'exposure': metrics.get('exposure_pct', 0.0),
                'knobs_json': json.dumps(policy),
            }
        )
        cursor += step_len

    with RESULTS_PATH.open('w', newline='') as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=[
                'start_idx','end_idx','regime_id','policy_source','pnl','max_dd','trades','avg_hold','exposure','knobs_json'
            ],
        )
        writer.writeheader()
        writer.writerows(rows)
