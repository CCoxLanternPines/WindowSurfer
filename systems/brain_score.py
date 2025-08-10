from __future__ import annotations

import csv
import json
import math
import sys
import time
from datetime import datetime
from pathlib import Path

from .brains import bear, bull, chop
from .brains._utils import first_hits
from .paths import brains_events_path, brains_summary_path


def _parse_horizon(h: str) -> int:
    h = h.strip().lower()
    if h.endswith("h"):
        return int(h[:-1])
    if h.endswith("d"):
        return int(h[:-1]) * 24
    return int(h)


def _load_series(tag: str, args=None) -> tuple[dict, datetime, datetime]:
    raw_dir = Path("data/raw")
    candidates = [raw_dir / f"{tag}.csv", raw_dir / f"{tag}.parquet", raw_dir / f"{tag.lower()}.csv"]
    if not any(p.exists() for p in candidates):
        base = tag
        for suff in ["USDT", "USD", "USDC"]:
            if tag.endswith(suff):
                base = tag[:-len(suff)]
                break
        candidates = [raw_dir / f"{base}.csv", raw_dir / f"{base}.parquet", raw_dir / f"{base.lower()}.csv"]
    for p in candidates:
        if p.exists():
            path = p
            break
    else:
        raise FileNotFoundError(f"no data for {tag}")
    if path.suffix == ".parquet":
        raise RuntimeError("parquet not supported without pandas")
    ts = []
    open_ = []
    high = []
    low = []
    close = []
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            ts.append(int(row["timestamp"]))
            open_.append(float(row["open"]))
            high.append(float(row["high"]))
            low.append(float(row["low"]))
            close.append(float(row["close"]))
    series = {"ts": ts, "open": open_, "high": high, "low": low, "close": close}
    if args is not None:
        start = datetime.fromisoformat(args.start) if args.start else datetime.fromtimestamp(ts[0])
        end = datetime.fromisoformat(args.end) if args.end else datetime.fromtimestamp(ts[-1])
    else:
        start = datetime.fromtimestamp(ts[0])
        end = datetime.fromtimestamp(ts[-1])
    return series, start, end


def _baseline(close, horizon):
    total = 0
    up = 0
    for i in range(len(close) - horizon):
        total += 1
        if close[i + horizon] > close[i]:
            up += 1
    return up / total if total else 0.0


def _wilson(p_hat, n, z=1.96):
    if n == 0:
        return 0.0, 0.0
    denom = 1 + z ** 2 / n
    center = p_hat + z ** 2 / (2 * n)
    adj = z * math.sqrt((p_hat * (1 - p_hat) + z ** 2 / (4 * n)) / n)
    low = (center - adj) / denom
    high = (center + adj) / denom
    return low, high


def _render_bar(i, n, brain, coin):
    width = 40
    done = int(width * i / max(1, n))
    bar = "#" * done + "-" * (width - done)
    sys.stdout.write(f"\r[{brain}:{coin}] {i}/{n} [{bar}]")
    sys.stdout.flush()


def _newline_after_bar():
    sys.stdout.write("\x1b[K\n")
    sys.stdout.flush()


def _score_events(events, baseline):
    hits = 0
    for i, e in enumerate(events, 1):
        hits += e["outcome"]
        p_hat = hits / i
        low, high = _wilson(p_hat, i)
        e.update({"p_hat": p_hat, "ci_low": low, "ci_high": high, "lift": p_hat - baseline})
    return events


def score_coin(coin, series, cfg, start, end, args):
    ts = series["ts"]
    close = series["close"]
    start_ts = int(start.timestamp())
    end_ts = int(end.timestamp())
    idx = [i for i, t in enumerate(ts) if start_ts <= t <= end_ts]
    if not idx:
        return []
    max_w = max(cfg["bear"].get("L", 0), cfg["chop"].get("S", 0), cfg["bull"].get("M", 0))
    horizon = max(
        _parse_horizon(cfg["bear"].get("horizons", ["24h"])[0]),
        _parse_horizon(cfg["chop"].get("horizons", ["24h"])[0]),
        _parse_horizon(cfg["bull"].get("horizons", ["24h"])[0]),
    )
    start_i = max(0, idx[0] - max_w)
    end_i = min(len(ts), idx[-1] + horizon + 1)
    ts = ts[start_i:end_i]
    close = close[start_i:end_i]
    series = {
        "ts": ts,
        "close": close,
        "open": series["open"][start_i:end_i],
        "high": series["high"][start_i:end_i],
        "low": series["low"][start_i:end_i],
    }
    idx = [i for i, t in enumerate(ts) if start_ts <= t <= end_ts]
    summaries = []
    brains = {
        "BEAR": (bear.parked, bear.parked_explain, cfg["bear"]),
        "CHOP": (chop.edge_long, chop.explain, cfg["chop"]),
        "BULL": (bull.momo_long, bull.explain, cfg["bull"]),
    }
    min_events = args.min_events if args.min_events is not None else cfg["scoring"].get("min_events", 100)
    for name, (func, expl, bcfg) in brains.items():
        horizon = _parse_horizon(bcfg.get("horizons", ["24h"])[0])
        if name == "BEAR":
            baseline_up = _baseline(close, horizon)
            baseline = 1.0 - baseline_up
        else:
            baseline = _baseline(close, horizon)
        events = []
        hits = 0
        total = idx[-1] if idx else 0
        if args.verbose >= 1:
            print(f"[{coin}] {name} horizon={horizon}h baseline={baseline*100:.1f}% (events so far: 0)")
        interrupted = False
        try:
            last_evt = -10**9
            for step, i in enumerate(idx):
                if i + horizon >= len(close):
                    break
                if name == "BEAR" and (i - last_evt) < horizon:
                    continue
                if args.verbose >= 2 and step % 1000 == 0:
                    p = hits / len(events) if events else 0.0
                    print(f"[{name}] i={i}/{total} events={len(events)} hits={hits} p={p*100:.1f}% lift={(p-baseline)*100:+.1f}%")
                if name == "BEAR":
                    dec = func(i, series, bcfg)
                    if args.verbose >= 3:
                        _, info = expl(i, series, bcfg)
                        reasons = info
                    else:
                        reasons = {}
                elif args.verbose >= 3:
                    res = expl(i, series, bcfg)
                    dec = res.get("decision")
                    reasons = res.get("reasons", {})
                else:
                    dec = func(i, series, bcfg)
                    reasons = {}
                if not dec:
                    continue
                if name == "BEAR":
                    last_evt = i
                    hit, _ = first_hits(close, i, bcfg.get("up_pct", 0.02), bcfg.get("down_pct", -0.02), horizon)
                    outcome = int(close[i + horizon] < close[i] or hit == "down")
                elif name == "CHOP":
                    hit, _ = first_hits(close, i, bcfg.get("tp_up", 0.04), bcfg.get("sl_dn", -0.03), horizon)
                    outcome = int(hit == "up")
                else:
                    hit, _ = first_hits(close, i, bcfg.get("tp_up", 0.06), bcfg.get("sl_dn", -0.04), horizon)
                    outcome = int(hit == "up")
                hits += outcome
                events.append({"ts": ts[i], "outcome": outcome})
                if args.verbose >= 2:
                    p = hits / len(events)
                    print(f"[{name}] i={i}/{total} events={len(events)} hits={hits} p={p*100:.1f}% lift={(p-baseline)*100:+.1f}%")
                if args.verbose >= 3:
                    t = datetime.fromtimestamp(ts[i]).isoformat(timespec="minutes")
                    parts = []
                    for k, v in reasons.items():
                        if isinstance(v, bool):
                            parts.append(f"{k}={'+' if v else '-'}")
                        elif isinstance(v, float):
                            parts.append(f"{k}={v:+.2f}")
                        else:
                            parts.append(f"{k}={v}")
                    reason_str = " ".join(parts)
                    print(f"[{name}] t={t} {reason_str} OK -> OUTCOME={'up' if outcome else 'down'}")
        except KeyboardInterrupt:
            interrupted = True
        events = _score_events(events, baseline)
        if not args.no_write and args.out:
            events_path = brains_events_path(coin, name)
            with open(events_path, "w", newline="") as f:
                writer = csv.DictWriter(
                    f,
                    fieldnames=["ts", "decision", "outcome", "p_hat", "ci_low", "ci_high", "lift"],
                )
                writer.writeheader()
                for e in events:
                    row = {
                        "ts": e["ts"],
                        "decision": 1,
                        "outcome": e["outcome"],
                        "p_hat": e["p_hat"],
                        "ci_low": e["ci_low"],
                        "ci_high": e["ci_high"],
                        "lift": e["lift"],
                    }
                    writer.writerow(row)
        p_hat = events[-1]["p_hat"] if events else 0.0
        ci_low, ci_high = _wilson(p_hat, len(events))
        lift = p_hat - baseline
        summaries.append(
            {
                "coin": coin,
                "brain": name,
                "events": len(events),
                "p_hat": p_hat,
                "ci_low": ci_low,
                "ci_high": ci_high,
                "baseline": baseline,
                "lift": lift,
                "insufficient_sample": int(len(events) < min_events),
                "drift": 0,
            }
        )
        if interrupted:
            break
    return summaries


def main(args):
    with open("settings.json") as f:
        cfg = json.load(f)["brains"]
    coins = [c.strip() for c in args.coins.split(",") if c.strip()]
    all_rows = []
    for coin in coins:
        series, start, end = _load_series(coin, args)
        all_rows.extend(score_coin(coin, series, cfg, start, end, args))
    if not args.no_write and args.out:
        summary_path = brains_summary_path()
        with open(summary_path, "w", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=["coin", "brain", "events", "p_hat", "ci_low", "ci_high", "baseline", "lift", "insufficient_sample", "drift"],
            )
            writer.writeheader()
            for row in all_rows:
                writer.writerow(row)
    else:
        print("coin brain events p_hat ci_low ci_high baseline lift")
        for row in all_rows:
            print(
                f"{row['coin']} {row['brain']} {row['events']} "
                f"{row['p_hat']*100:.1f}% {row['ci_low']*100:.1f}% {row['ci_high']*100:.1f}% "
                f"{row['baseline']*100:.1f}% {row['lift']*100:+.1f}%"
            )


def score_coin_single_brain(coin, series, cfg, start, end, brain_name, verbose, out=None):
    ts = series["ts"]
    close = series["close"]
    start_ts = int(start.timestamp())
    end_ts = int(end.timestamp())
    idx = [i for i, t in enumerate(ts) if start_ts <= t <= end_ts]
    if not idx:
        return []
    bcfg = cfg[brain_name.lower()]
    max_w = bcfg.get("L", 0)
    if brain_name == "CHOP":
        max_w = bcfg.get("S", 0)
    elif brain_name == "BULL":
        max_w = bcfg.get("M", 0)
    horizon = _parse_horizon(bcfg.get("horizons", ["24h"])[0])
    start_i = max(0, idx[0] - max_w)
    end_i = min(len(ts), idx[-1] + horizon + 1)
    ts = ts[start_i:end_i]
    close = close[start_i:end_i]
    series = {
        "ts": ts,
        "close": close,
        "open": series["open"][start_i:end_i],
        "high": series["high"][start_i:end_i],
        "low": series["low"][start_i:end_i],
    }
    idx = [i for i, t in enumerate(ts) if start_ts <= t <= end_ts]

    if brain_name == "BEAR":
        func = bear.parked
        expl = bear.parked_explain
        baseline_up = _baseline(close, horizon)
        baseline = 1.0 - baseline_up
    elif brain_name == "CHOP":
        func = chop.edge_long
        expl = chop.explain
        baseline = _baseline(close, horizon)
    else:
        func = bull.momo_long
        expl = bull.explain
        baseline = _baseline(close, horizon)

    events = []
    hits = 0
    total = len(idx)
    if verbose >= 1:
        print(f"[{coin}] {brain_name} horizon={horizon}h baseline={baseline*100:.1f}% (events so far: 0)")
    interrupted = False
    last_evt = -10**9
    try:
        for progress, i in enumerate(idx, 1):
            if i + horizon >= len(close):
                break
            if brain_name == "BEAR" and (i - last_evt) < horizon:
                _render_bar(progress, total, brain_name, coin)
                continue
            if brain_name == "BEAR":
                dec = func(i, series, bcfg)
                if verbose >= 3:
                    _, reasons = expl(i, series, bcfg)
                else:
                    reasons = {}
            else:
                if verbose >= 3:
                    res = expl(i, series, bcfg)
                    dec = res.get("decision")
                    reasons = res.get("reasons", {})
                else:
                    dec = func(i, series, bcfg)
                    reasons = {}
            if not dec:
                _render_bar(progress, total, brain_name, coin)
                continue
            if brain_name == "BEAR":
                last_evt = i
                hit, _ = first_hits(close, i, bcfg.get("up_pct", 0.02), bcfg.get("down_pct", -0.02), horizon)
                outcome = int(close[i + horizon] < close[i] or hit == "down")
            elif brain_name == "CHOP":
                hit, _ = first_hits(close, i, bcfg.get("tp_up", 0.04), bcfg.get("sl_dn", -0.03), horizon)
                outcome = int(hit == "up")
            else:
                hit, _ = first_hits(close, i, bcfg.get("tp_up", 0.06), bcfg.get("sl_dn", -0.04), horizon)
                outcome = int(hit == "up")
            hits += outcome
            events.append({"ts": ts[i], "outcome": outcome})
            if verbose >= 3:
                t = datetime.fromtimestamp(ts[i]).isoformat(timespec="minutes")
                parts = []
                for k in ["zL", "slopeM", "above_M", "reason"]:
                    if k in reasons:
                        v = reasons[k]
                        if isinstance(v, float):
                            parts.append(f"{k}={v:+.2f}")
                        else:
                            parts.append(f"{k}={v}")
                reason_str = " ".join(parts)
                outcome_str = "down" if outcome else "up"
                print(f"[{brain_name}] t={t} {reason_str} -> OUTCOME={outcome_str}")
                _render_bar(progress, total, brain_name, coin)
            else:
                _render_bar(progress, total, brain_name, coin)
    except KeyboardInterrupt:
        interrupted = True

    events = _score_events(events, baseline)
    if out:
        events_path = brains_events_path(coin, brain_name)
        with open(events_path, "w", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=["ts", "decision", "outcome", "p_hat", "ci_low", "ci_high", "lift"],
            )
            writer.writeheader()
            for e in events:
                row = {
                    "ts": e["ts"],
                    "decision": 1,
                    "outcome": e["outcome"],
                    "p_hat": e["p_hat"],
                    "ci_low": e["ci_low"],
                    "ci_high": e["ci_high"],
                    "lift": e["lift"],
                }
                writer.writerow(row)

    p_hat = events[-1]["p_hat"] if events else 0.0
    ci_low, ci_high = _wilson(p_hat, len(events))
    lift = p_hat - baseline
    summary = {
        "coin": coin,
        "brain": brain_name,
        "events": len(events),
        "p_hat": p_hat,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "baseline": baseline,
        "lift": lift,
        "insufficient_sample": int(len(events) < cfg["scoring"].get("min_events", 100)),
        "drift": 0,
    }

    if interrupted:
        _newline_after_bar()
        print([summary])

    return [summary]


def run_single_brain(args):
    with open("settings.json") as f:
        cfg = json.load(f)["brains"]
    coin = args.tag
    brain = args.brain.upper()
    series, start, end = _load_series(coin)
    summaries = score_coin_single_brain(
        coin, series, cfg, start, end, brain, args.verbose, args.out
    )
    _newline_after_bar()
    print(summaries)
