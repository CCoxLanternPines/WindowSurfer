from __future__ import annotations

"""Reader for simulation and live graph feeds."""

import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np


def discover_feed(
    *,
    mode: str,
    coin: str,
    account: Optional[str] = None,
    sim_dir: str = "data/temp/simulation",
    live_dir: str = "data/temp",
) -> Path:
    """Discover the latest feed path for simulation or live mode."""

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
    """Yield objects from an NDJSON file, optionally following new lines."""

    with path.open("r", encoding="utf-8") as fh:
        while True:
            line = fh.readline()
            if line:
                yield json.loads(line)
            elif follow:
                time.sleep(0.5)
            else:
                break


class GraphEngine:
    """Simple matplotlib engine showing trades with clickable markers."""

    def __init__(self, title: str = "Trade Feed") -> None:
        # layout: narrow info axis on left, main plot on right
        self.fig = plt.figure(figsize=(12, 6))
        gs = self.fig.add_gridspec(1, 2, width_ratios=[1, 4], wspace=0.05)
        self.info_ax = self.fig.add_subplot(gs[0, 0])
        self.ax_main = self.fig.add_subplot(gs[0, 1])

        # info panel setup
        self.info_ax.set_xticks([])
        self.info_ax.set_yticks([])
        for spine in self.info_ax.spines.values():
            spine.set_visible(False)
        self.info_text = self.info_ax.text(
            0.0,
            1.0,
            "Click a trade marker to view details",
            va="top",
            ha="left",
            transform=self.info_ax.transAxes,
        )
        self.info_text.set_fontfamily("monospace")
        self.info_text.set_fontsize(9)

        # main plot elements
        (self.price_line,) = self.ax_main.plot([], [], lw=1, label="Close")
        self.ax_main.grid(True)

        self.buy_trades: List[Dict[str, Any]] = []
        self.sell_trades: List[Dict[str, Any]] = []
        self.buy_art = self.ax_main.scatter(
            [], [], marker="^", c="green", edgecolor="black", picker=True, pickradius=6
        )
        self.sell_art = self.ax_main.scatter(
            [], [], marker="v", c="red", edgecolor="black", picker=True, pickradius=6
        )

        self.cursor_line = None

        self.fig.canvas.mpl_connect("pick_event", self._on_pick)
        self.ax_main.set_title(title)

        self.candle_x: List[float] = []
        self.candle_y: List[float] = []

    # ------------------------------------------------------------------
    def _on_pick(self, event) -> None:  # pragma: no cover - UI callback
        import json as _json

        if event.artist is self.buy_art:
            trade = self.buy_trades[int(event.ind[0])]
        elif event.artist is self.sell_art:
            trade = self.sell_trades[int(event.ind[0])]
        else:
            return

        txt = _json.dumps(trade, indent=2, sort_keys=True)
        self.info_text.set_text(txt)

        x = trade.get("i")
        if self.cursor_line is None:
            self.cursor_line = self.ax_main.axvline(x, color="gray", lw=0.8, ls="--")
        else:
            self.cursor_line.set_xdata(x)

        self.fig.canvas.draw_idle()

    # ------------------------------------------------------------------
    def add_object(self, obj: Dict[str, Any]) -> None:
        t = obj.get("t")
        if t == "c":
            self._add_candle(obj)
        elif t == "buy":
            self._add_trade(obj, is_buy=True)
        elif t == "sell":
            self._add_trade(obj, is_buy=False)

    # ------------------------------------------------------------------
    def _add_candle(self, candle: Dict[str, Any]) -> None:
        self.candle_x.append(candle.get("i"))
        self.candle_y.append(candle.get("c"))
        self.price_line.set_data(self.candle_x, self.candle_y)
        self.ax_main.relim()
        self.ax_main.autoscale_view()
        self.fig.canvas.draw_idle()

    # ------------------------------------------------------------------
    def _add_trade(self, trade: Dict[str, Any], *, is_buy: bool) -> None:
        artist = self.buy_art if is_buy else self.sell_art
        trades = self.buy_trades if is_buy else self.sell_trades
        trades.append(trade)

        x = trade.get("i")
        y = trade.get("p")
        offsets = artist.get_offsets()
        if offsets.size:
            new_offsets = np.vstack([offsets, [x, y]])
        else:
            new_offsets = np.array([[x, y]])
        artist.set_offsets(new_offsets)
        self.fig.canvas.draw_idle()


def render_feed(path: Path, follow: bool = False) -> None:
    """Render a feed file and optionally follow new data."""

    engine = GraphEngine(title=path.name)

    if follow:
        plt.ion()

    for obj in _stream(path, follow):
        engine.add_object(obj)
        if follow:
            plt.pause(0.01)

    if not follow:
        plt.show()

