from __future__ import annotations

"""Candle-by-candle price visualizer."""

from pathlib import Path
import sys

import pandas as pd
import matplotlib
matplotlib.use("TkAgg")  # or "Qt5Agg" if you have Qt installed
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
    total_frames = (N - start_idx + frameskip - 1) // frameskip
    
    zoom_frames = int(total_frames * 1)  # zoom done at 40% of draw time

    x_full = (float(t[0]), float(t[-1]))
    y_full = (float(y.min()), float(y.max()))

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

    # Outside update(), initialize before FuncAnimation:
    x_zoom_start_frame = None
    y_zoom_start_frame = None

    def update(frame: int):
        nonlocal x_zoom_start_frame, y_zoom_start_frame
        idx = min(start_idx + frame * frameskip, N - 1)
        xs.append(float(t[idx]))
        ys.append(float(y[idx]))
        line.set_data(xs, ys)
        if xs and ys:
            dot.set_data([xs[-1]], [ys[-1]])

        # Detect milestones
        if y_zoom_start_frame is None and min(ys) <= y_full[0]:
            y_zoom_start_frame = frame
        if x_zoom_start_frame is None and xs[0] <= x_full[0]:
            x_zoom_start_frame = frame

        # Progress for each axis
        # X
        if x_zoom_start_frame is not None:
            
            u_x = min(max((frame - x_zoom_start_frame) / zoom_frames, 0.0), 1.0)
            s_x = u_x * u_x * (3 - 2 * u_x)
            xlim = _lerp((xs[0], xs[-1]), x_full, s_x)
        else:
            xlim = (xs[0], xs[-1])

        # Y
        if y_zoom_start_frame is not None:
            u_y = min(max((frame - y_zoom_start_frame) / zoom_frames, 0.0), 1.0)
            s_y = u_y * u_y * (3 - 2 * u_y)
            current_min = min(ys)
            current_max = max(ys)
            ylim = _pad(_lerp((current_min, current_max), y_full, s_y), 0.02)
        else:
            ylim = _pad((min(ys), max(ys)), 0.02)

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
