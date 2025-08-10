from __future__ import annotations

from systems.data_loader import load_or_fetch
from systems.scripts.evaluate_buy import evaluate_buy
from systems.scripts.evaluate_sell import evaluate_sell
from systems.scripts.execute_buy import execute_buy
from systems.scripts.execute_sell import execute_sell
from systems.scripts import ledger
import numpy as np


def _max_drawdown(equity: np.ndarray) -> float:
    """Compute maximum drawdown for an equity curve."""
    if equity.size == 0:
        return 0.0
    peaks = np.maximum.accumulate(equity)
    drawdowns = (equity - peaks) / np.where(peaks == 0, 1.0, peaks)
    return float(-np.min(drawdowns)) if drawdowns.size else 0.0


def run_sim_blocks(tag: str, ranges, knobs: dict, verbose: int = 0):
    """Run simulation over ``ranges`` using production evaluate/execute paths.

    Parameters
    ----------
    tag : str
        Asset identifier for candle loading.
    ranges : iterable
        Iterable of ``(start, end)`` tuples where each element may be a
        timestamp or candle index.
    knobs : dict
        Parameter knobs forwarded to evaluation/execute functions.
    verbose : int, optional
        Verbosity level; ``>0`` enables audit logging.

    Returns
    -------
    dict
        Summary containing ``pnl``, ``maxdd`` and ``trades``.
    """
    # parity audit: ensure we know which runner is active
    from systems.sim_engine import RUNNER_ID

    if verbose:
        print(f"[AUDIT] Sim runner active: {RUNNER_ID} (production parity mode)")

    candles = load_or_fetch(tag)
    ts = candles["timestamp"].to_numpy()
    sim_ledger = ledger.new_ledger()
    equity_curve: list[float] = []

    for start, end in ranges:
        if isinstance(start, (int, np.integer)) and isinstance(end, (int, np.integer)) and start <= end:
            i0, i1 = int(start), int(end)
        else:
            i0 = int(np.searchsorted(ts, start, side="left"))
            i1 = int(np.searchsorted(ts, end, side="right") - 1)
        block_df = candles.iloc[i0:i1+1]

        for _, candle in block_df.iterrows():
            sell_signal = evaluate_sell(candle, sim_ledger, knobs, verbose)
            if sell_signal:
                execute_sell(candle, sim_ledger, knobs, verbose, sim_mode=True)

            buy_signal = evaluate_buy(candle, sim_ledger, knobs, verbose)
            if buy_signal:
                execute_buy(candle, sim_ledger, knobs, verbose, sim_mode=True)

            equity_curve.append(
                ledger.current_equity(sim_ledger, float(candle["close"]))
            )

    pnl = float(ledger.total_pnl(sim_ledger))
    maxdd = _max_drawdown(np.asarray(equity_curve, dtype=float))
    trades = int(ledger.trade_count(sim_ledger))
    return {"pnl": pnl, "maxdd": maxdd, "trades": trades}


__all__ = ["run_sim_blocks"]
