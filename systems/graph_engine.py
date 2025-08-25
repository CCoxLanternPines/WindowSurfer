from __future__ import annotations

"""Reader for simulation and live graph feeds."""

import json
import time
from pathlib import Path
from typing import Dict, List, Any, Optional

import matplotlib.pyplot as plt
import pandas as pd


def discover_feed(
    *,
    mode: str,
    coin: str,
    account: Optional[str] = None,
    sim_dir: str = "data/temp/simulation",
    live_dir: str = "data/temp",
) -> Path:
    coin = coin.replace("/", "").upper()
    if mode == "sim":
        base = Path(sim_dir)
        pattern = f"{coin}_*.json"
        files = sorted(base.glob(pattern))
        if not files:
            raise FileNotFoundError(f"No simulation feed found for {coin} in {base}")
        return files[-1]
    if not account:
        raise ValueError("account required for live mode")
    path = Path(live_dir) / f"{account}_{coin}.json"
    if not path.exists():
        raise FileNotFoundError(f"Feed not found: {path}")
    return path


def _stream(path: Path, follow: bool = False):
    with path.open("r", encoding="utf-8") as fh:
        while True:
            line = fh.readline()
            if line:
                yield json.loads(line)
            elif follow:
                time.sleep(0.5)
            else:
                break


def render_feed(path: Path, follow: bool = False) -> None:
    candles: List[Dict[str, Any]] = []
    buys: List[Dict[str, Any]] = []
    sells: List[Dict[str, Any]] = []
    for obj in _stream(path, follow):
        t = obj.get("t")
        if t == "c":
            candles.append(obj)
        elif t == "buy":
            buys.append(obj)
        elif t == "sell":
            sells.append(obj)

    if not candles:
        return

    df = pd.DataFrame(candles)
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(df["i"], df["c"], lw=1, label="Close")
    for b in buys:
        ax.scatter(b["i"], b["p"], marker="^", c="green", edgecolor="black")
    for s in sells:
        ax.scatter(s["i"], s["p"], marker="v", c="red", edgecolor="black")
    ax.set_title(path.name)
    ax.grid(True)
    plt.show()
