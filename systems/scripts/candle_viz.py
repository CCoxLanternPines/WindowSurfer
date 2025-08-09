from __future__ import annotations

"""Candle-by-candle price visualizer."""

from pathlib import Path
import sys

import numpy as np
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
    return df[["timestamp", "high", "low", "close"]]


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
    *,
    k: int = 50,
    s: int = 20,
    m: int = 50,
    g_thr: float = 0.25,
    d_min: float = 1.2,
    q: int = 8,
    squeeze_thr: float = 0.7,
    s_thr: float = 1e-4,
    i_thr: float = 0.6,
    t_thr: float = 1.5,
) -> None:
    """Render a candle-by-candle price visualizer with overlays."""

    df = load_raw(tag)
    t = df["timestamp"].to_numpy()
    c = df["close"].to_numpy()
    h = df["high"].to_numpy()
    l = df["low"].to_numpy()
    N = len(c)
    if start_idx >= N:
        print("[ERROR] start index beyond data length")
        return

    # ------------------------------------------------------------------
    # Precompute series
    # ------------------------------------------------------------------
    r = np.diff(c, prepend=c[0])
    ema_k = pd.Series(c).ewm(span=k, adjust=False).mean().to_numpy()

    prev_close = np.r_[c[0], c[:-1]]
    tr = np.maximum.reduce([
        h - l,
        np.abs(h - prev_close),
        np.abs(l - prev_close),
    ])
    atr = pd.Series(tr).ewm(span=k, adjust=False).mean().to_numpy()

    pos = np.maximum(r, 0)
    neg = np.maximum(-r, 0)
    U = pd.Series(pos).ewm(span=s, adjust=False).mean().to_numpy()
    D = pd.Series(neg).ewm(span=s, adjust=False).mean().to_numpy()
    G = (U - D) / (U + D + 1e-9)
    min_g = np.minimum.accumulate(G)
    GD = G - min_g

    I = r / (atr + 1e-9)
    T = (c - ema_k) / (atr + 1e-9)
    atr_ma = pd.Series(atr).ewm(span=s, adjust=False).mean().to_numpy()
    atr_ratio = atr / (atr_ma + 1e-9)
    ema_m = pd.Series(c).ewm(span=m, adjust=False).mean().to_numpy()
    ema_m_slope = np.gradient(ema_m, t)

    tsnl = np.zeros(N, dtype=int)
    min_price = c[0]
    for i in range(1, N):
        if c[i] <= min_price:
            min_price = c[i]
            tsnl[i] = 0
        else:
            tsnl[i] = tsnl[i - 1] + 1

    S = (
        (GD >= d_min)
        & (tsnl >= q)
        & (atr_ratio <= squeeze_thr)
        & (np.abs(ema_m_slope) <= s_thr)
    )
    prev_S = np.r_[False, S[:-1]]
    thrust = prev_S & (I >= i_thr) & (T >= t_thr)

    fps = max(1, int(1000 / max(1, speed_ms)))
    zoom_frames = max(fps * zoom_seconds, 1)

    x_full = (float(t[0]), float(t[-1]))
    y_full = (float(c.min()), float(c.max()))

    fig, (ax, ax_g) = plt.subplots(
        2,
        1,
        sharex=True,
        figsize=(width, height),
        gridspec_kw={"height_ratios": [3, 1]},
    )

    line, = ax.plot([], [], lw=1)
    center_line, = ax.plot([], [], lw=1, color="C1")
    upper_band, = ax.plot([], [], lw=1, color="C2", alpha=0.5)
    lower_band, = ax.plot([], [], lw=1, color="C2", alpha=0.5)
    dot, = ax.plot([], [], "o", ms=4)
    stall_scatter, = ax.plot([], [], "o", ms=4, color="orange")
    thrust_scatter, = ax.plot([], [], "o", ms=4, color="green")
    if show_grid:
        ax.grid(True, alpha=0.2)
        ax_g.grid(True, alpha=0.2)

    title_left = ax.text(0.01, 0.99, tag, transform=ax.transAxes, ha="left", va="top")
    title_right = ax.text(0.99, 0.99, "", transform=ax.transAxes, ha="right", va="top")

    g_line, = ax_g.plot([], [], lw=1, color="C0")
    g_fill = ax_g.fill_between([], [], 0, color="C0", alpha=0.1)
    ax_g.axhline(0, color="k", lw=0.5)
    ax_g.set_ylim(-1, 1)

    xs: list[float] = []
    ys: list[float] = []
    stall_x: list[float] = []
    stall_y: list[float] = []
    thrust_x: list[float] = []
    thrust_y: list[float] = []

    x_zoom_start: int | None = None
    y_zoom_start: int | None = None
    x_start_lim: tuple[float, float] | None = None
    y_start_lim: tuple[float, float] | None = None

    def on_key(event):  # pragma: no cover - UI interaction
        if event.key == "escape":
            plt.close(fig)

    fig.canvas.mpl_connect("key_press_event", on_key)

    def update(frame: int):
        nonlocal g_fill, x_zoom_start, y_zoom_start, x_start_lim, y_start_lim
        idx = min(start_idx + frame * frameskip, N - 1)
        x_i = float(t[idx])
        y_i = float(c[idx])
        xs.append(x_i)
        ys.append(y_i)
        line.set_data(xs, ys)
        dot.set_data([x_i], [y_i])

        center_line.set_data(t[: idx + 1], ema_k[: idx + 1])
        upper_band.set_data(t[: idx + 1], ema_k[: idx + 1] + atr[: idx + 1])
        lower_band.set_data(t[: idx + 1], ema_k[: idx + 1] - atr[: idx + 1])

        if S[idx]:
            stall_x.append(x_i)
            stall_y.append(y_i)
        if thrust[idx]:
            thrust_x.append(x_i)
            thrust_y.append(y_i)
        stall_scatter.set_data(stall_x, stall_y)
        thrust_scatter.set_data(thrust_x, thrust_y)

        g_line.set_data(t[: idx + 1], G[: idx + 1])
        if g_fill:
            g_fill.remove()
        g_fill = ax_g.fill_between(
            t[: idx + 1],
            G[: idx + 1],
            0,
            where=G[: idx + 1] < 0,
            color="C0",
            alpha=0.1,
        )

        x_cur = (xs[0], xs[-1])
        y_cur = (min(ys), max(ys))

        if x_zoom_start is None and x_cur[0] <= x_full[0]:
            x_zoom_start = frame
            x_start_lim = x_cur
        if y_zoom_start is None and y_cur[0] <= y_full[0]:
            y_zoom_start = frame
            y_start_lim = y_cur

        if x_zoom_start is not None and x_start_lim is not None:
            u_x = min(max((frame - x_zoom_start) / zoom_frames, 0.0), 1.0)
            s_x = u_x * u_x * (3 - 2 * u_x)
            xlim = _lerp(x_start_lim, x_full, s_x)
        else:
            xlim = x_cur

        if y_zoom_start is not None and y_start_lim is not None:
            u_y = min(max((frame - y_zoom_start) / zoom_frames, 0.0), 1.0)
            s_y = u_y * u_y * (3 - 2 * u_y)
            ylim = _pad(_lerp(y_start_lim, y_full, s_y), 0.02)
        else:
            ylim = _pad(y_cur, 0.02)

        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        ax_g.set_xlim(*xlim)

        price = y_i
        ts = pd.to_datetime(x_i, unit="s")
        title_right.set_text(f"{price:.4f} @ {ts}")
        return (
            line,
            dot,
            center_line,
            upper_band,
            lower_band,
            stall_scatter,
            thrust_scatter,
            g_line,
            g_fill,
            title_right,
        )

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
