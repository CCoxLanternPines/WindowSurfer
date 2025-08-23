from __future__ import annotations

import csv
import importlib
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any, Dict, List

import matplotlib.pyplot as plt
import pandas as pd

from systems.brains.base import default_init


# ---------------------------------------------------------------------------
# Utilities (trimmed versions referencing legacy for structure)
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]


def load_settings() -> Dict[str, Any]:
    """Load settings JSON, trying project settings then legacy copy."""
    candidates = [PROJECT_ROOT / "settings/settings.json", PROJECT_ROOT / "legacy/settings/settings.json"]
    for path in candidates:
        if path.exists():
            with path.open("r", encoding="utf-8") as fh:
                return json.load(fh)
    raise FileNotFoundError("settings.json not found in settings/ or legacy/settings/")


def load_ledger_config(ledger_name: str) -> Dict[str, Any]:
    settings = load_settings()
    ledgers = settings.get("ledger_settings", {})
    if ledger_name not in ledgers:
        raise ValueError(f"Ledger '{ledger_name}' not found in settings")
    return ledgers[ledger_name]


def resolve_ccxt_symbols(settings: Dict[str, Any], ledger: str) -> tuple[str, str]:
    cfg = settings.get("ledger_settings", {}).get(ledger, {})
    return cfg.get("kraken_name", ""), cfg.get("binance_name", "")


def to_tag(symbol: str) -> str:
    return symbol.replace("/", "").replace(":", "").upper()


def sim_path_csv(tag: str) -> str:
    data_dir = PROJECT_ROOT / "data/sim"
    p1 = data_dir / f"{tag}_1h.csv"
    if p1.exists():
        return str(p1)
    return str(data_dir / f"{tag}.csv")


def parse_cutoff(value: str) -> timedelta:
    value = value.strip().lower()
    num = int(value[:-1])
    unit = value[-1]
    if unit == "h":
        return timedelta(hours=num)
    if unit == "d":
        return timedelta(days=num)
    if unit == "w":
        return timedelta(weeks=num)
    if unit == "m":
        return timedelta(days=30 * num)
    if unit == "y":
        return timedelta(days=365 * num)
    raise ValueError("cutoff unit must be one of h,d,w,m,y")


# ---------------------------------------------------------------------------
# Core brain engine
# ---------------------------------------------------------------------------

def run_brain(
    *,
    ledger: str,
    brain_name: str,
    time_window: str | None = None,
    verbose: int | bool = 0,
    viz: bool = False,
) -> Dict[str, Any] | None:
    settings = load_settings()
    ledger_cfg = load_ledger_config(ledger)
    kraken_symbol, _ = resolve_ccxt_symbols(settings, ledger)
    tag = to_tag(kraken_symbol)
    csv_path = sim_path_csv(tag)
    path = Path(csv_path)
    if not path.exists():
        print(f"[ERROR] Missing data file: {csv_path}")
        raise SystemExit(1)

    df = pd.read_csv(csv_path)

    # Timestamp normalization
    ts_col: str | None = None
    for c in df.columns:
        lc = str(c).lower()
        if lc in ("timestamp", "time", "date"):
            ts_col = c
            break
    if ts_col is None:
        raise ValueError(f"No timestamp column in {csv_path}")

    df[ts_col] = pd.to_numeric(df[ts_col], errors="coerce")
    df = df.dropna(subset=[ts_col])
    before = len(df)
    df = df.sort_values(ts_col).drop_duplicates(subset=[ts_col], keep="last").reset_index(drop=True)
    removed = before - len(df)
    if not df[ts_col].is_monotonic_increasing:
        raise ValueError(f"Candles not sorted by {ts_col}: {csv_path}")

    first_ts = int(df[ts_col].iloc[0]) if len(df) else None
    last_ts = int(df[ts_col].iloc[-1]) if len(df) else None
    first_iso = (
        datetime.fromtimestamp(first_ts, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        if first_ts is not None
        else "n/a"
    )
    last_iso = (
        datetime.fromtimestamp(last_ts, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        if last_ts is not None
        else "n/a"
    )
    print(
        f"[DATA] file={csv_path} rows={len(df)} first={first_iso} last={last_iso} dups_removed={removed}"
    )

    now = datetime.now(tz=timezone.utc)
    cutoff_ts = None
    start_from = "full"
    if time_window:
        try:
            delta = parse_cutoff(time_window)
            candidate_cutoff = now.timestamp() - delta.total_seconds()
        except Exception:
            try:
                dt = datetime.fromisoformat(time_window.replace("Z", "+00:00"))
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=timezone.utc)
                else:
                    dt = dt.astimezone(timezone.utc)
                candidate_cutoff = dt.timestamp()
            except Exception as exc:
                print(f"[ERROR] Invalid --time value: {time_window}")
                raise SystemExit(1) from exc
        if first_ts is not None and candidate_cutoff < first_ts:
            print("[BRAIN][TIME] cutoff before first candle -> using full history.")
        else:
            cutoff_ts = candidate_cutoff
            start_from = datetime.fromtimestamp(
                cutoff_ts, tz=timezone.utc
            ).strftime("%Y-%m-%dT%H:%M:%SZ")
    now_iso = now.strftime("%Y-%m-%dT%H:%M:%SZ")
    print(f"[BRAIN][TIME] mode=brain start_from={start_from} now_utc={now_iso}")
    if cutoff_ts is not None:
        df = df[df[ts_col] >= cutoff_ts].reset_index(drop=True)
        rows_after = len(df)
        print(f"[BRAIN][TIME] applied cutoff={start_from} rows_after={rows_after}")
        if rows_after == 0:
            print("[ABORT][BRAIN][TIME] No candles ≥ cutoff")
            return None

    brain = importlib.import_module(f"systems.brains.{brain_name}")
    state = (
        brain.init(settings, ledger_cfg)
        if hasattr(brain, "init")
        else default_init(settings, ledger_cfg)
    )

    events: List[Dict[str, Any]] = []
    for t in range(len(df)):
        evs = brain.tick(t, df, state)
        for e in evs:
            ts_val = int(df.iloc[t][ts_col]) if ts_col else None
            price = float(df.iloc[t]["close"])
            iso = (
                datetime.fromtimestamp(ts_val, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
                if ts_val is not None
                else ""
            )
            rec: Dict[str, Any] = {
                "t": t,
                "time": ts_val,
                "time_iso": iso,
                "price": price,
                "type": e.get("type", ""),
            }
            if "score" in e:
                rec["score"] = float(e["score"])
            extra = {k: v for k, v in e.items() if k not in {"type", "score", "t"}}
            if extra:
                rec["extra"] = extra
            events.append(rec)

    counts = Counter(e["type"] for e in events)
    latencies: Dict[str, float] = {}
    by_type: Dict[str, List[int]] = defaultdict(list)
    for e in events:
        by_type[e["type"]].append(e["t"])
    for typ, idxs in by_type.items():
        if len(idxs) > 1:
            gaps = [idxs[i + 1] - idxs[i] for i in range(len(idxs) - 1)]
            latencies[typ] = sum(gaps) / len(gaps)
    extras: Dict[str, Any] = {"latencies": latencies}
    if hasattr(brain, "summarize"):
        try:
            extras.update(brain.summarize(events, df, state) or {})
        except Exception:
            pass

    print(f"[BRAIN] counts={dict(counts)}")

    ts_tag = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    logs_dir = PROJECT_ROOT / "logs"
    logs_dir.mkdir(exist_ok=True)
    report_path = logs_dir / f"brain_report_{ledger}_{brain_name}_{ts_tag}.json"
    events_path = logs_dir / f"brain_events_{ledger}_{brain_name}_{ts_tag}.csv"
    with report_path.open("w", encoding="utf-8") as fh:
        json.dump({"counts": dict(counts), "events": events, "extras": extras}, fh, indent=2)
    with events_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["time_iso", "price", "type", "score"])
        for e in events:
            writer.writerow([e.get("time_iso"), e.get("price"), e.get("type"), e.get("score", "")])

    plot_path = None
    if viz:
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(df.index, df["close"], label="Price", color="blue", lw=1)
        markers = {"buy": "^", "sell": "v", "flag": "o", "note": "x"}
        colors = {"buy": "green", "sell": "red", "flag": "orange", "note": "purple"}
        for typ in counts:
            xs = [e["t"] for e in events if e["type"] == typ]
            ys = [e["price"] for e in events if e["type"] == typ]
            ax.scatter(xs, ys, marker=markers.get(typ, "o"), c=colors.get(typ, "black"), label=typ)
        ax.legend()
        tmp_dir = PROJECT_ROOT / "data/tmp"
        tmp_dir.mkdir(parents=True, exist_ok=True)
        plot_path = tmp_dir / f"brain_plot_{ledger}_{brain_name}_{ts_tag}.png"
        fig.savefig(plot_path)
        plt.show()

    return {
        "counts": dict(counts),
        "extras": extras,
        "report_path": str(report_path),
        "events_path": str(events_path),
        "plot_path": str(plot_path) if plot_path else None,
    }
