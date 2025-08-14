"""Interactive sliding candlestick viewer with pluggable math modules."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import importlib.util
from pathlib import Path
from typing import Callable, List

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.widgets import Button, Slider
import pandas as pd

from _stepper_io import read_candles, resolve_csv

MODULES_PATH = Path(__file__).resolve().parent.parent / "math_modules"
DEFAULT_WINDOW = 200


@dataclass
class Module:
    name: str
    lookback: int
    func: Callable[[pd.DataFrame, int], dict[str, float | int | None]]


def load_modules() -> List[Module]:
    """Dynamically import all modules from math_modules/ directory."""
    modules: List[Module] = []
    for path in MODULES_PATH.glob("*.py"):
        if path.name.startswith("_"):
            continue
        spec = importlib.util.spec_from_file_location(path.stem, path)
        if not spec or not spec.loader:
            continue
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        name = getattr(module, "NAME", None)
        lookback = getattr(module, "LOOKBACK", None)
        calc = getattr(module, "calculate", None)
        if name and isinstance(lookback, int) and callable(calc):
            modules.append(Module(name, lookback, calc))
    return modules


def precompute(df: pd.DataFrame) -> pd.DataFrame:
    """Precompute rolling indicators used by modules."""
    df["close_pct"] = df["close"].pct_change()
    hl = df["high"] - df["low"]
    hc = (df["high"] - df["close"].shift()).abs()
    lc = (df["low"] - df["close"].shift()).abs()
    tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)
    df["atr_5"] = tr.rolling(5).mean()
    df["atr_20"] = tr.rolling(20).mean()
    return df


def display_metrics(ax: plt.Axes, metrics: dict[str, float | int | None]) -> None:
    """Render metric text items on the metrics axis."""
    ax.clear()
    ax.axis("off")
    for idx, (name, value) in enumerate(metrics.items()):
        ax.text(
            0.01,
            1 - idx * 0.07,
            f"{name}: {value}",
            transform=ax.transAxes,
            va="top",
            fontsize=9,
        )


class Candles:
    """Maintain candle artists for fast updates."""

    def __init__(self, ax: plt.Axes):
        self.ax = ax
        self.wicks: list[plt.Line2D] = []
        self.bodies: list[Rectangle] = []
        self.target_line = ax.axvline(0, color="grey", linestyle="--", visible=False)

    def draw(self, df: pd.DataFrame) -> None:
        while len(self.wicks) < len(df):
            (line,) = self.ax.plot([], [], color="black")
            rect = Rectangle((0, 0), 0, 0, edgecolor="black")
            self.ax.add_patch(rect)
            self.wicks.append(line)
            self.bodies.append(rect)

        xs = mdates.date2num(df["timestamp"].to_numpy())
        width = (xs[1] - xs[0]) * 0.8 if len(xs) > 1 else 0.6

        for i, row in enumerate(df.itertuples(index=False)):
            x = xs[i]
            open_, high, low, close = row.open, row.high, row.low, row.close
            self.wicks[i].set_data([x, x], [low, high])
            lower = min(open_, close)
            height = abs(close - open_)
            body = self.bodies[i]
            body.set_xy((x - width / 2, lower))
            body.set_width(width)
            body.set_height(height if height != 0 else 0.001)
            body.set_facecolor("green" if close >= open_ else "red")
            body.set_visible(True)
        for j in range(len(df), len(self.wicks)):
            self.wicks[j].set_data([], [])
            self.bodies[j].set_visible(False)

        self.ax.set_xlim(xs[0], xs[-1])
        self.ax.set_ylim(df["low"].min(), df["high"].max())

    def mark_target(self, ts: pd.Timestamp | None) -> None:
        if ts is None:
            self.target_line.set_visible(False)
        else:
            x = mdates.date2num(ts)
            self.target_line.set_xdata([x, x])
            self.target_line.set_visible(True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("tag", nargs="?", help="Symbol tag or CSV path")
    parser.add_argument("--csv", help="Direct CSV path")
    parser.add_argument("--window", type=int, default=DEFAULT_WINDOW, help="Window size")
    args = parser.parse_args()

    csv_path = resolve_csv(args.tag, args.csv)
    print(f"Using CSV: {csv_path}")
    df = read_candles(csv_path)
    df = precompute(df)

    modules = load_modules()

    fig = plt.figure(figsize=(12, 6))
    gs = fig.add_gridspec(2, 2, height_ratios=[4, 1], width_ratios=[3, 1])
    ax_candles = fig.add_subplot(gs[0, 0])
    ax_metrics = fig.add_subplot(gs[0, 1])
    ax_slider = fig.add_subplot(gs[1, 0])
    ax_metrics.axis("off")

    candles = Candles(ax_candles)

    max_index = len(df) - 2
    slider = Slider(ax_slider, "Index", 0, max_index, valinit=min(args.window, max_index), valstep=1)

    def update_idx(idx: int) -> None:
        idx = max(0, min(int(idx), max_index))
        slider.eventson = False
        slider.set_val(idx)
        slider.eventson = True
        start = max(0, idx - args.window + 1)
        view = df.iloc[start : idx + 1]
        candles.draw(view)
        target_ts = df["timestamp"].iloc[idx + 1] if idx + 1 < len(df) else None
        candles.mark_target(target_ts)
        metrics: dict[str, float | int | None] = {}
        df_slice = df.iloc[: idx + 1]
        for mod in modules:
            if idx >= mod.lookback:
                result = mod.func(df_slice, idx)
                for k, v in result.items():
                    metrics[f"{mod.name} {k}"] = v
            else:
                metrics[mod.name] = None
        display_metrics(ax_metrics, metrics)
        fig.canvas.draw_idle()

    def on_slider(val: float) -> None:
        update_idx(int(round(val)))

    slider.on_changed(on_slider)

    # Buttons
    btn_ax_play = fig.add_axes([0.55, 0.05, 0.05, 0.04])
    btn_ax_back = fig.add_axes([0.61, 0.05, 0.05, 0.04])
    btn_ax_fwd = fig.add_axes([0.67, 0.05, 0.05, 0.04])
    btn_ax_bigback = fig.add_axes([0.73, 0.05, 0.05, 0.04])
    btn_ax_bigfwd = fig.add_axes([0.79, 0.05, 0.05, 0.04])

    btn_play = Button(btn_ax_play, "\u25B6")  # ▶
    btn_back = Button(btn_ax_back, "\u23EE")  # ⏮
    btn_fwd = Button(btn_ax_fwd, "\u23ED")  # ⏭
    btn_bigback = Button(btn_ax_bigback, "\u23EA")  # ⏪
    btn_bigfwd = Button(btn_ax_bigfwd, "\u23E9")  # ⏩

    playing = {"value": False}
    timer = fig.canvas.new_timer(interval=200)

    def play_step() -> None:
        if slider.val >= max_index:
            playing["value"] = False
            return
        update_idx(int(slider.val) + 1)
        if playing["value"]:
            timer.start()

    timer.add_callback(play_step)

    def toggle_play(event) -> None:  # noqa: D401 - callback signature
        playing["value"] = not playing["value"]
        if playing["value"]:
            timer.start()

    def step(delta: int) -> None:
        update_idx(int(slider.val) + delta)

    btn_play.on_clicked(toggle_play)
    btn_back.on_clicked(lambda event: step(-1))
    btn_fwd.on_clicked(lambda event: step(1))
    btn_bigback.on_clicked(lambda event: step(-args.window // 2))
    btn_bigfwd.on_clicked(lambda event: step(args.window // 2))

    def on_key(event) -> None:
        if event.key == "left":
            step(-1)
        elif event.key == "right":
            step(1)
        elif event.key == "up":
            step(args.window // 10)
        elif event.key == "down":
            step(-args.window // 10)
        elif event.key == "home":
            update_idx(0)
        elif event.key == "end":
            update_idx(max_index)

    fig.canvas.mpl_connect("key_press_event", on_key)

    update_idx(int(slider.val))
    plt.show()


if __name__ == "__main__":
    main()
