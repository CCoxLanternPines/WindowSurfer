from __future__ import annotations

import importlib
import inspect
from typing import Any, Iterable

# candidate modules to search for a sim runner
_CANDIDATE_MODULES: list[str] = [
    "systems.scripts.simple_sim_engine",
    "systems.simple_sim_engine",
    "systems.scripts.sim_engine",
    "engine.simple_sim_engine",
    "engine.sim_engine",
]

# candidate function names within those modules
_CANDIDATE_FUNCTIONS: list[str] = [
    "run_sim_blocks",
    "run_simulation_blocks",
    "run_sim",
    "run_simulation",
]


def _resolve_runner() -> tuple[Any, str, Any]:
    """Return (module, fn_name, fn) for the first available runner."""
    last_err: Exception | None = None
    for mod_name in _CANDIDATE_MODULES:
        try:
            mod = importlib.import_module(mod_name)
        except Exception as e:  # pragma: no cover - import errors are aggregated
            last_err = e
            continue
        for fn_name in _CANDIDATE_FUNCTIONS:
            if hasattr(mod, fn_name):
                return mod, fn_name, getattr(mod, fn_name)
    search = [(m, _CANDIDATE_FUNCTIONS) for m in _CANDIDATE_MODULES]
    raise ImportError(
        f"[sim_shim] No sim runner found. Tried: {search}\nLast error: {last_err}"
    )


_MOD, _FN_NAME, _RUNNER = _resolve_runner()
RUNNER_ID = f"{_MOD.__name__}.{_FN_NAME}"


def _call_runner_with_ranges(tag: str, ranges: Iterable, knobs: dict, verbose: int):
    params = inspect.signature(_RUNNER).parameters
    # handle low-level simple engines expecting candles+blocks
    if "candles" in params and "blocks" in params:
        from .data_loader import load_or_fetch  # lazy import
        import numpy as np

        candles = load_or_fetch(tag)
        ts = candles["timestamp"].to_numpy()
        blocks = []
        for i, (start, end) in enumerate(ranges, 1):
            if isinstance(start, (int, np.integer)) and isinstance(end, (int, np.integer)) and start <= end:
                i0, i1 = int(start), int(end)
            else:
                i0 = int(np.searchsorted(ts, start, side="left"))
                i1 = int(np.searchsorted(ts, end, side="right") - 1)
            blocks.append({"test_index_start": i0, "test_index_end": i1, "block_id": i})
        res = _RUNNER(candles, blocks, knobs)
        if isinstance(res, dict):
            return res
        summary = getattr(res, "summary", None)
        return summary if isinstance(summary, dict) else res

    kwargs = {"tag": tag, "knobs": knobs, "verbose": verbose}
    if "ranges" in params:
        kwargs["ranges"] = ranges
    elif "block_ranges" in params:
        kwargs["block_ranges"] = ranges
    elif "start_end_ranges" in params:
        kwargs["start_end_ranges"] = ranges
    else:  # assume positional (tag, ranges, knobs, verbose)
        return _RUNNER(tag, ranges, knobs, verbose)
    return _RUNNER(**kwargs)


def run_sim_blocks(tag, ranges, knobs, verbose: int = 0):
    """
    ranges: list of (start_ts, end_ts) or list of candle-index tuples.
    Must return dict with at least: {'pnl': float, 'maxdd': float, 'trades': int}
    """
    if _FN_NAME in ("run_sim_blocks", "run_simulation_blocks"):
        return _call_runner_with_ranges(tag, ranges, knobs, verbose)

    # runner handles a single block; aggregate over ranges
    total_pnl = 0.0
    total_trades = 0
    equity = 0.0
    peak = 0.0
    maxdd = 0.0
    for start, end in ranges:
        params = inspect.signature(_RUNNER).parameters
        kwargs = {"tag": tag, "knobs": knobs, "verbose": verbose}
        if "resume" in params:
            kwargs["resume"] = True
        if "start" in params:
            kwargs["start"] = start
        elif "start_ts" in params:
            kwargs["start_ts"] = start
        elif "i0" in params:
            kwargs["i0"] = start
        if "end" in params:
            kwargs["end"] = end
        elif "end_ts" in params:
            kwargs["end_ts"] = end
        elif "i1" in params:
            kwargs["i1"] = end
        try:
            res = _RUNNER(**kwargs)
        except TypeError:
            res = _RUNNER(tag, start, end, knobs, verbose)
        pnl = float(res.get("pnl", 0.0))
        total_pnl += pnl
        total_trades += int(res.get("trades", 0))
        equity += pnl
        peak = max(peak, equity)
        if peak > 0:
            maxdd = max(maxdd, (peak - equity) / peak)
        block_dd = float(res.get("maxdd", res.get("max_dd", 0.0)))
        maxdd = max(maxdd, block_dd)
    return {"pnl": float(total_pnl), "maxdd": float(maxdd), "trades": int(total_trades)}


__all__ = ["run_sim_blocks", "RUNNER_ID"]
