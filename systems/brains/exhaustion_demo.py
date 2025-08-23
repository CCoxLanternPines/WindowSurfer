from __future__ import annotations

import pandas as pd


def init(settings, ledger_cfg):
    return {"short": 12, "long": 36}


def tick(t: int, df: pd.DataFrame, state):
    out = []
    if t == 0:
        return out
    s, l = state["short"], state["long"]
    if "ma_s" not in state:
        state["ma_s"] = df["close"].rolling(s).mean()
        state["ma_l"] = df["close"].rolling(l).mean()
    if t < max(s, l):
        return out
    prev_up = state["ma_s"].iloc[t - 1] > state["ma_l"].iloc[t - 1]
    now_up = state["ma_s"].iloc[t] > state["ma_l"].iloc[t]
    if (not prev_up) and now_up:
        out.append({"type": "buy", "score": float(state["ma_s"].iloc[t] - state["ma_l"].iloc[t])})
    if prev_up and (not now_up):
        out.append({"type": "sell", "score": float(state["ma_l"].iloc[t] - state["ma_s"].iloc[t])})
    return out


def summarize(events, df, state):
    from collections import Counter

    c = Counter(e["type"] for e in events)
    return {"counts": dict(c)}
