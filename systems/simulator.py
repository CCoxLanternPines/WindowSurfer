from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Iterable, Dict, Any, List

import numpy as np
import pandas as pd

from systems.data_loader import load_or_fetch
from systems.paths import log_file
from systems.brain import RegimeBrain
from systems.features import extract_features, ALL_FEATURES
from systems.policy_blender import (
    load_seed_knobs,
    classify_current,
    predict_next,
    apply_hysteresis,
    blend_knobs,
)

try:  # pragma: no cover - fallbacks if scripts package missing
    from systems.scripts.evaluate_buy import evaluate_buy  # type: ignore
    from systems.scripts.evaluate_sell import evaluate_sell  # type: ignore
except Exception:  # pragma: no cover
    def evaluate_buy(*args, **kwargs):
        return False

    def evaluate_sell(*args, **kwargs):
        return False

RUNNER_ID = "systems.simulator.run_sim_blocks"


# -----------------------
# Small utils
# -----------------------

def _finite(x: Any, fallback: float = 0.0) -> float:
    """Coerce any value to a finite float; otherwise return fallback."""
    try:
        xf = float(x)
    except Exception:
        return fallback
    return xf if math.isfinite(xf) else fallback


def _clean_prices(df: pd.DataFrame) -> pd.DataFrame:
    """Replace non-finite OHLC with last finite value; drop rows if all NaN."""
    out = df.copy()
    for col in ("open", "high", "low", "close"):
        if col in out:
            s = pd.to_numeric(out[col], errors="coerce").astype(float)
            s = s.replace([np.inf, -np.inf], np.nan).ffill()
            # if first values are NaN, forward fill won't help; backfill once
            s = s.bfill()
            out[col] = s
    # Drop rows still containing NaNs in close (must have a tradeable price)
    out = out[~out["close"].isna()]
    return out


# -----------------------
# Indicators / risk
# -----------------------

def _rsi(series: np.ndarray, period: int = 14) -> np.ndarray:
    delta = np.diff(series, prepend=series[0])
    up = np.where(delta > 0, delta, 0.0)
    dn = np.where(delta < 0, -delta, 0.0)
    au = pd.Series(up).ewm(alpha=1 / period, adjust=False).mean().to_numpy()
    ad = pd.Series(dn).ewm(alpha=1 / period, adjust=False).mean().to_numpy()
    rs = np.divide(au, np.maximum(ad, 1e-12))
    rsi = 100 - (100 / (1 + rs))
    return rsi


def _max_drawdown(equity: np.ndarray) -> float:
    """Max DD as a positive fraction. Handles empty or non-finite gracefully."""
    if equity.size == 0:
        return 0.0
    arr = np.array(equity, dtype=float)
    # Replace non-finite with previous finite (or first finite), else flatten
    if not np.isfinite(arr[0]):
        finite_first = np.nanmin(np.where(np.isfinite(arr), arr, np.nan))
        if not np.isfinite(finite_first):
            return 0.0
        arr[0] = finite_first
    mask = ~np.isfinite(arr)
    if mask.any():
        arr[mask] = np.interp(np.flatnonzero(mask), np.flatnonzero(~mask), arr[~mask])
    peaks = np.maximum.accumulate(arr)
    peaks = np.where(peaks == 0.0, 1e-12, peaks)
    dd = 1.0 - (arr / peaks)
    dd[~np.isfinite(dd)] = 0.0
    return float(np.max(dd))


# -----------------------
# One-block sim
# -----------------------

def _run_block(
    df: pd.DataFrame,
    knobs: Dict[str, Any],
    equity0: float,
    *,
    verbosity: int = 0,
    blend_enabled: bool = False,
    brain: RegimeBrain | None = None,
    seed_knobs: Dict[str, Dict[str, Any]] | None = None,
    feat_idx: List[int] | None = None,
    alpha: float = 0.7,
    hyst_boost: float = 0.1,
) -> Dict[str, Any]:
    df = _clean_prices(df)
    if df.empty:
        # No candles → flat result
        return {
            "pnl": 0.0,
            "maxdd": 0.0,
            "trades": 0,
            "equity_series": [float(equity0)],
            "returns": [0.0],
            "ledger": {"cash": float(equity0)},
        }

    close = df["close"].to_numpy(dtype=float)
    high = df["high"].to_numpy(dtype=float)
    low = df["low"].to_numpy(dtype=float)
    ts = df["timestamp"].to_numpy(dtype=int)

    rsi = _rsi(close, period=14)

    # Blending context
    history: List[int] = []
    last_top: int | None = None
    win = 50

    # Seed equity series at initial value (prevents empty/NaN issues)
    equity = np.full(len(close), float(equity0), dtype=float)
    cash = float(equity0)
    positions: List[Dict[str, float]] = []
    buy_cd = 0
    sell_cd = 0
    trades = 0
    ticks_in_pos = 0
    closed_rets: list[float] = []

    for i in range(len(close)):
        price = float(close[i])
        timestamp = ts[i]

        current_knobs = knobs
        if (
            blend_enabled
            and brain is not None
            and seed_knobs is not None
            and i >= win
        ):
            window = df.iloc[i - win : i]
            feats = extract_features(window)
            window_feats = feats[feat_idx] if feat_idx is not None else feats
            weights_now = classify_current(window_feats, brain)
            weights_next = predict_next(weights_now, history, alpha)
            weights_final = apply_hysteresis(weights_next, last_top, hyst_boost)
            current_knobs = blend_knobs(weights_final, seed_knobs)
            if verbosity >= 3:
                print(
                    f"[BLEND] now={weights_now.round(3).tolist()} "
                    f"next={weights_next.round(3).tolist()} "
                    f"final={weights_final.round(3).tolist()} "
                    f"knobs={current_knobs}"
                )
            top = int(np.argmax(weights_final))
            history.append(top)
            if len(history) > 50:
                history.pop(0)
            last_top = top

        rsi_buy = int(current_knobs.get("rsi_buy", 35))
        rsi_sell = int(current_knobs.get("rsi_sell", 65))
        buy_cooldown_cfg = int(current_knobs.get("buy_cooldown", 8))
        sell_cooldown_cfg = int(current_knobs.get("sell_cooldown", 8))
        take_profit = float(current_knobs.get("take_profit", 0.03))
        stop_loss = float(current_knobs.get("stop_loss", 0.06))
        trailing_stop = float(current_knobs.get("trailing_stop", 0.02))
        position_pct = float(current_knobs.get("position_pct", 0.06))
        max_concurrent = int(current_knobs.get("max_concurrent", 2))

        if verbosity >= 3:
            print(
                f"[TICK] {pd.to_datetime(timestamp, unit='s', utc=True)} "
                f"close={price:.2f} open_notes={len(positions)} "
                f"cool(buy={buy_cd},sell={sell_cd})",
            )

        for p in positions:
            p["peak"] = max(p["peak"], price)

        if sell_cd == 0 and positions:
            keep: List[Dict[str, float]] = []
            for p in positions:
                entry = p["entry"]
                ret = (price - entry) / max(entry, 1e-12)
                trail_hit = (p["peak"] > 0.0) and ((p["peak"] - price) / p["peak"] >= trailing_stop)
                if (ret >= take_profit) or (ret <= -stop_loss) or trail_hit or (rsi[i] >= rsi_sell):
                    cash += p["size"] * (1.0 + ret)
                    trades += 1
                    closed_rets.append(ret)
                    if verbosity >= 3:
                        print(
                            f"[SELL] note amt={p['size']/entry:.6f} at={price:.2f} pnl=${(p['size']*ret):.2f}"
                        )
                else:
                    keep.append(p)
            positions = keep
            sell_cd = sell_cooldown_cfg if trades else 0
        else:
            sell_cd = max(0, sell_cd - 1)

        if buy_cd == 0 and len(positions) < max_concurrent and rsi[i] <= rsi_buy:
            size = cash * position_pct
            if size > 0.0:
                positions.append({"entry": price, "size": size, "peak": price})
                cash -= size
                trades += 1
                buy_cd = buy_cooldown_cfg
                if verbosity >= 3:
                    print(f"[BUY] note amt={size/max(price,1e-12):.6f} at={price:.2f}")
        else:
            buy_cd = max(0, buy_cd - 1)

        pos_val = 0.0
        for p in positions:
            entry = max(p["entry"], 1e-12)
            pos_val += p["size"] * (price / entry)
        equity[i] = _finite(cash + pos_val, fallback=equity[i-1] if i > 0 else equity0)
        if positions:
            ticks_in_pos += 1
    # Force close remaining positions at last price
    if positions:
        price = float(close[-1])
        for p in positions:
            entry = max(p["entry"], 1e-12)
            ret = (price - entry) / entry
            cash += p["size"] * (1.0 + ret)
            closed_rets.append(ret)
        positions.clear()

    equity[-1] = _finite(cash, fallback=equity[-1])

    pnl = _finite(equity[-1] - equity0, 0.0)
    maxdd = _max_drawdown(equity)

    # Returns: pct change per step relative to previous equity (not equity0)
    returns = np.zeros_like(equity, dtype=float)
    prev = np.where(equity[:-1] <= 0.0, 1e-12, equity[:-1])
    returns[1:] = (equity[1:] - equity[:-1]) / prev

    block_time_in_market = ticks_in_pos / max(len(close), 1)
    block_win_ct = sum(1 for r in closed_rets if r > 0)
    block_win_rate = block_win_ct / max(len(closed_rets), 1)

    return {
        "pnl": pnl,
        "maxdd": maxdd,
        "trades": int(trades),
        "equity_series": equity.tolist(),
        "returns": returns.tolist(),
        "ledger": {"cash": float(cash)},
        "time_in_market": float(block_time_in_market),
        "win_rate": float(block_win_rate),
        "closed_rets": closed_rets,
        "had_trades": bool(len(closed_rets) > 0),
    }


# -----------------------
# Multi-block orchestrator
# -----------------------

def run_sim_blocks(
    blocks: Iterable[Dict[str, Any]],
    tag: str,
    knobs: Dict[str, Any],
    settings: Dict[str, Any],
    *,
    verbosity: int = 0,
    run_id: str | None = None,
    log_path: str | None = None,
    audit: bool = True,
    blend_enabled: bool = False,
    alpha: float = 0.7,
    hyst_boost: float = 0.1,
) -> Dict[str, Any]:
    """Execute simulation over blocks."""

    # UTF-8 logs (Windows-safe)
    log_fh = None
    if run_id:
        path = Path(log_path) if log_path else log_file(tag, run_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        log_fh = path.open("a", encoding="utf-8", newline="")

    def _log(msg: str, log_fh=log_fh) -> None:
        safe = msg.encode("utf-8", "replace").decode("utf-8")
        print(safe)
        if log_fh is not None:
            log_fh.write(safe + "\n")
            log_fh.flush()

    _log(f"[SIM] Using sim runner: {RUNNER_ID}")
    if audit:
        _log(f"[AUDIT] Sim runner active: {RUNNER_ID} (production parity mode)")

    sim_dir = Path(f"data/tmp/simulation/{tag}")
    if sim_dir.exists():
        for p in sim_dir.glob("*.json"):
            p.unlink()
    else:
        sim_dir.mkdir(parents=True, exist_ok=True)

    candles = load_or_fetch(tag)
    # Ensure timestamps are int64 seconds
    candles = candles.copy()
    candles["timestamp"] = pd.to_numeric(candles["timestamp"], errors="coerce").astype("Int64")
    candles = candles.dropna(subset=["timestamp"]).astype({"timestamp": "int64"})
    ts = candles["timestamp"].to_numpy()

    # Starting capital
    capital = _finite(settings.get("capital", 1000.0), 1000.0)

    brain = None
    seed_knobs_by_regime: Dict[str, Dict[str, Any]] | None = None
    feat_idx: List[int] | None = None
    if blend_enabled:
        brain_path = Path("data/brains") / f"brain_{tag}.json"
        brain = RegimeBrain.from_file(brain_path)
        feat_order = brain._b.get("features", [])
        feat_idx = [ALL_FEATURES.index(f) for f in feat_order]
        seed_all = load_seed_knobs()
        seed_knobs_by_regime = seed_all.get(tag, {})
        if not seed_knobs_by_regime:
            raise ValueError(f"No seed knobs for tag {tag}")
        if not knobs and seed_knobs_by_regime:
            knobs = next(iter(seed_knobs_by_regime.values()))
    equity_curve: List[float] = []
    total_trades = 0
    total_pnl = 0.0
    time_in_market_sum = 0.0
    candles_sum = 0
    all_closed_rets: List[float] = []
    blocks_traded = 0

    block_count = 0
    for k, block in enumerate(blocks, 1):
        block_count = k
        if "i0" in block and "i1" in block:
            i0, i1 = int(block["i0"]), int(block["i1"])
        else:
            start = block.get("start_ts") or block.get("start")
            end = block.get("end_ts") or block.get("end")
            i0 = int(np.searchsorted(ts, int(start), side="left")) if start is not None else 0
            i1 = int(np.searchsorted(ts, int(end), side="right") - 1) if end is not None else len(ts) - 1

        block_df = candles.iloc[max(i0, 0): min(i1 + 1, len(candles))].copy()
        if block_df.empty:
            # Still log the block boundary if verbose
            if verbosity >= 2:
                _log(f"[BLOCK] k={k:02d} | empty block | candles=0")
            continue

        res = _run_block(
            block_df,
            knobs,
            capital,
            verbosity=verbosity,
            blend_enabled=blend_enabled,
            brain=brain,
            seed_knobs=seed_knobs_by_regime,
            feat_idx=feat_idx,
            alpha=alpha,
            hyst_boost=hyst_boost,
        )
        capital += _finite(res.get("pnl"), 0.0)
        total_trades += int(res.get("trades", 0))
        total_pnl += _finite(res.get("pnl"), 0.0)
        time_in_market_sum += res.get("time_in_market", 0.0) * len(block_df)
        candles_sum += len(block_df)
        all_closed_rets.extend(res.get("closed_rets", []))
        if res.get("had_trades"):
            blocks_traded += 1

        es = res.get("equity_series") or []
        if es:
            equity_curve.extend([_finite(x, capital) for x in es])

        # Per-block ledger snapshot (minimal, but keeps parity hooks)
        ledger_path = sim_dir / f"block_{k:02d}.json"
        try:
            with ledger_path.open("w", encoding="utf-8") as fh:
                json.dump(res.get("ledger", {}), fh, indent=2)
        except Exception:
            # Never crash on IO during tuning
            pass

        if verbosity >= 2:
            start_dt = pd.to_datetime(block_df.iloc[0]["timestamp"], unit="s", utc=True)
            end_dt = pd.to_datetime(block_df.iloc[-1]["timestamp"], unit="s", utc=True)
            _log(f"[BLOCK] k={k:02d} | {start_dt.date()} -> {end_dt.date()} | candles={len(block_df)}")
            _log(
                f"[BLOCK] trades={int(res.get('trades',0))} "
                f"pnl=${_finite(res.get('pnl')):.2f} "
                f"maxdd={_finite(res.get('maxdd')):.2f} "
                f"tim={_finite(res.get('time_in_market')):.2f} "
                f"win={_finite(res.get('win_rate')):.2f}"
            )

    # Ensure equity_curve has at least one point
    if not equity_curve:
        equity_curve = [float(capital)]

    maxdd_all = _max_drawdown(np.asarray(equity_curve, dtype=float))
    time_in_market = time_in_market_sum / max(candles_sum, 1)
    time_flat = 1.0 - time_in_market
    win_rate = (sum(1 for r in all_closed_rets if r > 0) /
                max(len(all_closed_rets), 1))
    avg_trade_ret = float(np.mean(all_closed_rets)) if all_closed_rets else 0.0

    result = {
        "trades": int(total_trades),
        "pnl": _finite(total_pnl, 0.0),
        "maxdd": _finite(maxdd_all, 0.0),
        "equity_series": equity_curve,
        "time_in_market": float(time_in_market),
        "time_flat": float(time_flat),
        "win_rate": float(win_rate),
        "avg_trade_ret": float(avg_trade_ret),
        "blocks_traded": int(blocks_traded),
    }

    if verbosity >= 1:
        pnl_dd = result["pnl"] * (1 - 1.5 * result["maxdd"])
        _log(
            f"[TRIAL] tag={tag} blocks={block_count} trades={result['trades']} "
            f"pnl=${result['pnl']:,.2f} maxdd={result['maxdd']:.2f} pnl_dd=${pnl_dd:,.2f} "
            f"| tim={time_in_market:.2f} win={win_rate:.2f} btr={blocks_traded}"
        )

    if log_fh is not None:
        try:
            log_fh.close()
        except Exception:
            pass

    return result



__all__ = ["run_sim_blocks", "RUNNER_ID"]
