"""Interactive candlestick stepper with pluggable math modules."""

from __future__ import annotations

import argparse
import importlib.util
import pathlib
from typing import Callable, Dict

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

try:
    import mplfinance as mpf
    HAS_MPLFINANCE = True
except Exception:  # pragma: no cover - optional dependency
    HAS_MPLFINANCE = False

MODULES_PATH = pathlib.Path(__file__).resolve().parent.parent / "math_modules"
DEFAULT_WINDOW = 100


def load_modules() -> Dict[str, Callable[[pd.DataFrame, int], dict[str, float]]]:
    """Dynamically import all calculate functions from math_modules."""
    modules: Dict[str, Callable[[pd.DataFrame, int], dict[str, float]]] = {}
    for path in MODULES_PATH.glob("*.py"):
        if path.name.startswith("_"):
            continue
        spec = importlib.util.spec_from_file_location(path.stem, path)
        if spec and spec.loader:
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            calc = getattr(module, "calculate", None)
            if callable(calc):
                modules[path.stem] = calc
    return modules


def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure standard OHLCV column names and types."""
    df = df.rename(columns={c: c.strip().lower() for c in df.columns})
    df = df[["timestamp", "open", "high", "low", "close", "volume"]]
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df


def plot_segment(ax: plt.Axes, segment: pd.DataFrame, tag: str) -> None:
    """Plot a candle segment on given axis."""
    ax.clear()
    if HAS_MPLFINANCE:
        mpf.plot(
            segment.set_index("timestamp"),
            type="candle",
            ax=ax,
            style="yahoo",
            warn_too_much_data=9999999,
            show_nontrading=True,
        )
    else:  # fallback line plot
        ax.plot(segment["timestamp"], segment["close"], color="black")
    ax.set_title(f"{tag} @ index {segment.index[-1]}")


def display_metrics(ax: plt.Axes, metrics: dict[str, float]) -> None:
    """Render metric text items on the metrics axis."""
    ax.clear()
    ax.axis("off")
    for idx, (name, value) in enumerate(metrics.items()):
        color = "green" if isinstance(value, (int, float)) and value >= 0 else "red"
        ax.text(
            0.01,
            1 - idx * 0.1,
            f"{name}: {value}",
            transform=ax.transAxes,
            va="top",
            fontsize=9,
            color=color,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("tag", help="Symbol tag corresponding to data/raw/<TAG>.csv")
    parser.add_argument(
        "--window", type=int, default=DEFAULT_WINDOW, help="Number of candles to display"
    )
    args = parser.parse_args()

    csv_path = pathlib.Path("data/raw") / f"{args.tag}.csv"
    df = pd.read_csv(csv_path)
    df = standardize_columns(df)

    modules = load_modules()

    fig = plt.figure(figsize=(10, 6))
    gs = fig.add_gridspec(2, 2, height_ratios=[4, 1], width_ratios=[3, 1])
    ax_candles = fig.add_subplot(gs[0, 0])
    ax_metrics = fig.add_subplot(gs[0, 1])
    ax_slider = fig.add_subplot(gs[1, 0])
    ax_metrics.axis("off")

    slider = Slider(ax_slider, "Index", 0, len(df) - 1, valinit=args.window, valstep=1)

    def update(val: float) -> None:
        idx = int(val)
        start = max(0, idx - args.window)
        segment = df.iloc[start : idx + 1]
        plot_segment(ax_candles, segment, args.tag)
        metrics: dict[str, float] = {}
        for name, fn in modules.items():
            try:
                metrics.update(fn(df, idx))
            except Exception as exc:  # pragma: no cover - runtime errors
                metrics[name] = f"error: {exc}"
        display_metrics(ax_metrics, metrics)
        fig.canvas.draw_idle()

    slider.on_changed(update)
    update(slider.val)
    plt.show()


if __name__ == "__main__":
    main()
