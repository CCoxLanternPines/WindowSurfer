from __future__ import annotations

from typing import Callable, Dict, Any

import numpy as np


def run_sim(
    *,
    prices: np.ndarray,
    base_settings: Dict[str, Any],
    policy_provider: Callable[[int], Dict[str, Any]],
    start_idx: int,
    end_idx: int,
    fees_bps: float,
    slip_bps: float,
) -> Dict[str, Any]:
    """Very small placeholder simulation.

    The engine simply applies a buy-and-hold on the slice between ``start_idx``
    and ``end_idx``. The ``policy_provider`` is invoked for the block start but
    otherwise unused. Fees and slippage are subtracted from the theoretical
    profit for demonstration purposes.
    """
    capital = float(base_settings.get('capital', 0))
    if end_idx > len(prices):
        end_idx = len(prices)
    start_price = prices[start_idx]
    end_price = prices[end_idx - 1]
    gross = capital * (end_price / start_price - 1)
    cost = capital * (fees_bps + slip_bps)
    pnl = gross - cost
    return {
        'final_capital': capital + pnl,
        'pnl': pnl,
        'max_dd': 0.0,
        'trades': 1,
        'avg_hold': end_idx - start_idx,
        'exposure_pct': 1.0,
        'regime_id_used': policy_provider(start_idx).get('regime', '')
    }
