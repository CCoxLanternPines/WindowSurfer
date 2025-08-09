from __future__ import annotations

"""Candle-by-candle price visualizer."""

from pathlib import Path
import sys

import pandas as pd
import matplotlib
# Use a non-interactive backend by default for headless environments
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter, PillowWriter

try:
    from systems.utils.config import resolve_path
except Exception:  # pragma: no cover - fallback if config not present
    def resolve_path(rel_path: str) -> Path:
        return Path(__file__).resolve().parents[2] / rel_path


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_raw(tag: str) -> pd.DataFrame:
    """Load raw candle data for ``tag``.

    Parameters
    ----------
    tag: str
        Trading pair tag such as ``DOGEUSD``.
    """

    base_path = resolve_path("data/raw")
    csv_path = base_path / f"{tag}.csv"
    if not csv_path.exists():
        print(f"[ERROR] Data file not found: {csv_path}")
        sys.exit(1)

    df = pd.read_csv(csv_path, usecols=["timestamp", "open", "high", "low", "close", "volume"])
    df = df.sort_values("timestamp").reset_index(drop=True)
    return df[["timestamp", "close"]]


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def _pad(lim: tuple[float, float], pct: float) -> tuple[float, float]:
    span = lim[1] - lim[0]
    return lim[0] - span * pct, lim[1] + span * pct


def _lerp(a: tuple[float, float], b: tuple[float, float], s: float) -> tuple[float, float]:
    return a[0] + (b[0] - a[0]) * s, a[1] + (b[1] - a[1]) * s


def run_price_viz(
    tag: str,
    speed_ms: int = 10,
    frameskip: int = 1,
    start_idx: int = 0,
    zoom_seconds: int = 5,
    width: float = 12.0,
    height: float = 6.0,
    save_path: str | None = None,
    show_grid: bool = False,
) -> None:
    """Render a candle-by-candle price visualizer."""

    df = load_raw(tag)
    t = df["timestamp"].to_numpy()
    y = df["close"].to_numpy()
    N = len(y)
    if start_idx >= N:
        print("[ERROR] start index beyond data length")
        return

    fps = max(1, int(1000 / max(1, speed_ms)))
    zoom_frames = max(fps * zoom_seconds, 1)

    x_full = (float(t[0]), float(t[-1]))
    y_full = (float(y.min()), float(y.max()))

    end_init = min(start_idx + 200, N - 1)
    x0 = (float(t[start_idx]), float(t[end_init]))
    y0 = (
        float(y[start_idx : end_init + 1].min()),
        float(y[start_idx : end_init + 1].max()),
    )

    fig, ax = plt.subplots(figsize=(width, height))
    line, = ax.plot([], [], lw=1)
    dot, = ax.plot([], [], "o", ms=4)
    if show_grid:
        ax.grid(True, alpha=0.2)

    title_left = ax.text(0.01, 0.99, tag, transform=ax.transAxes, ha="left", va="top")
    title_right = ax.text(0.99, 0.99, "", transform=ax.transAxes, ha="right", va="top")

    xs: list[float] = []
    ys: list[float] = []

    def on_key(event):  # pragma: no cover - UI interaction
        if event.key == "escape":
            plt.close(fig)

    fig.canvas.mpl_connect("key_press_event", on_key)

    def update(frame: int):
        idx = min(start_idx + frame * frameskip, N - 1)
        xs.append(float(t[idx]))
        ys.append(float(y[idx]))
        line.set_data(xs, ys)
        dot.set_data(xs[-1], ys[-1])

        u = min(max(frame / zoom_frames, 0.0), 1.0)
        s = u * u * (3 - 2 * u)
        xlim = _lerp(x0, x_full, s)
        ylim = _pad(_lerp(y0, y_full, s), 0.02)
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)

        price = ys[-1]
        ts = pd.to_datetime(xs[-1], unit="s")
        title_right.set_text(f"{price:.4f} @ {ts}")
        return line, dot, title_right

    total_frames = (N - start_idx + frameskip - 1) // frameskip
    anim = FuncAnimation(
        fig,
        update,
        frames=range(total_frames),
        interval=speed_ms,
        blit=True,
    )

    if save_path:
        save_path = str(save_path)
        ext = Path(save_path).suffix.lower()
        writer = None
        if ext == ".mp4":
            if matplotlib.animation.writers.is_available("ffmpeg"):
                writer = FFMpegWriter(fps=fps)
            else:
                print("ffmpeg not available, falling back to GIF")
                save_path = str(Path(save_path).with_suffix(".gif"))
                writer = PillowWriter(fps=fps)
        else:
            writer = PillowWriter(fps=fps)
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        anim.save(save_path, writer=writer)
        plt.close(fig)
    else:
        plt.show()
