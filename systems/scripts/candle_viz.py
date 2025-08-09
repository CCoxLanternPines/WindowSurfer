from __future__ import annotations

"""Candle-by-candle price visualizer with zoom-at-wall behavior and overlays."""

from pathlib import Path
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("TkAgg")  # Change to "Qt5Agg" if you use Qt
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter, PillowWriter
from matplotlib.patches import Rectangle


try:
    from systems.utils.config import resolve_path
except Exception:  # fallback if config not present
    def resolve_path(rel_path: str) -> Path:
        return Path(__file__).resolve().parents[2] / rel_path


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_raw(tag: str) -> pd.DataFrame:
    """Load raw candle data for `tag`."""
    base_path = resolve_path("data/raw")
    csv_path = base_path / f"{tag}.csv"
    if not csv_path.exists():
        print(f"[ERROR] Data file not found: {csv_path}")
        sys.exit(1)

    df = pd.read_csv(csv_path, usecols=["timestamp", "open", "high", "low", "close", "volume"])
    df = df.sort_values("timestamp").reset_index(drop=True)
    return df[["timestamp", "high", "low", "close"]]


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

def _pad(lim: tuple[float, float], pct: float) -> tuple[float, float]:
    span = lim[1] - lim[0]
    return lim[0] - span * pct, lim[1] + span * pct


def _lerp(a: tuple[float, float], b: tuple[float, float], s: float) -> tuple[float, float]:
    return a[0] + (b[0] - a[0]) * s, a[1] + (b[1] - a[1]) * s


# ---------------------------------------------------------------------------
# Main visualizer
# ---------------------------------------------------------------------------

def run_price_viz(
    tag: str,
    speed_ms: int = 10,
    frameskip: int = 1,
    start_idx: int = 0,
    width: float = 12.0,
    height: float = 6.0,
    save_path: str | None = None,
    show_grid: bool = False,
    zoom_seconds: int | None = None,   # <-- add this back for compatibility
    *,
    k: int = 50, s: int = 20, m: int = 50,
    g_thr: float = 0.25, d_min: float = 1.2, q: int = 8,
    squeeze_thr: float = 0.7, s_thr: float = 1e-4,
    i_thr: float = 0.6, t_thr: float = 1.5,
    snap_on: bool = True, snap_mode: str = "low", snap_alpha: float = 0.9,
    snap_center_mult: int = 3, snap_fill: bool = True,
) -> None:

    df = load_raw(tag)
    t = df["timestamp"].to_numpy()
    c = df["close"].to_numpy()
    h = df["high"].to_numpy()
    l = df["low"].to_numpy()
    all_time_low = np.nanmin(c)
    all_time_high = np.nanmax(c)
    N = len(c)
    if start_idx >= N:
        print("[ERROR] start index beyond data length")
        return

    window_size = 200  # number of candles to keep in view

    # ------------------------------------------------------------------
    # Precompute series (vectorized)
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

    rolling_low_s = pd.Series(c).rolling(window=s, min_periods=1).min().to_numpy()
    if snap_mode == "ema":
        snap = np.maximum(0.0, (c - ema_k)) / (atr + 1e-9)
    else:  # "low"
        snap = (c - rolling_low_s) / (atr + 1e-9)

    snap_avg = (
        pd.Series(snap).rolling(window=window_size, min_periods=1).mean().to_numpy()
    )

    snap_center_span = max(snap_center_mult * s, s + 1)
    snap_center = pd.Series(snap).ewm(span=snap_center_span, adjust=False).mean().to_numpy()

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

    # Animation timing
    total_frames = (N - start_idx + frameskip - 1) // frameskip
    fps = max(1, int(1000 / max(1, speed_ms)))
    zoom_frames = int(total_frames * 1)  # same pacing as previous commit

    # Full-series extents
    x_full = (float(t[0]), float(t[-1]))
    y_full = (float(c.min()), float(c.max()))

    # Figure with gravity subplot
    fig, (ax, ax_g) = plt.subplots(
        2, 1,
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

    # All-time range bar
    bar_ax = ax.twinx()
    bar_ax.set_ylim(all_time_low, all_time_high)
    bar_ax.set_xlim(0, 1)
    bar_ax.yaxis.set_ticks_position("right")
    bar_ax.yaxis.set_label_position("right")
    bar_ax.set_yticklabels([])
    bar_ax.set_xticklabels([])

    bar_width = 0.2
    bar_bg = Rectangle(
        (0.4, all_time_low),
        bar_width,
        all_time_high - all_time_low,
        facecolor="lightgray",
        alpha=0.3,
    )
    bar_ax.add_patch(bar_bg)

    bar_marker = Rectangle(
        (0.4, np.nan),
        bar_width,
        (all_time_high - all_time_low) * 0.01,
        facecolor="red",
        alpha=0.8,
    )
    bar_ax.add_patch(bar_marker)

    title_left = ax.text(0.01, 0.99, tag, transform=ax.transAxes, ha="left", va="top")
    title_right = ax.text(0.99, 0.99, "", transform=ax.transAxes, ha="right", va="top")

    status_text = ax.text(
        0.01,
        0.01,
        "",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        color="0.3",
    )

    def refresh_hud():
        # show current speed, inventory and realized PnL
        title_right.set_text(
            f"{state['price']:.4f}  spd:{speed_ms_cur}ms  inv:{len(holdings)}  "
            f"buys:{buys} sells:{sells}  PnL:{score:+.2f}"
        )
        fig.canvas.draw_idle()

    blue_line, = ax_g.plot([], [], lw=1.0, color="blue", alpha=0.8)
    ax_g.axhline(0, color="k", lw=0.5)
    ax_g.set_ylim(-1, 1)

    if snap_on:
        ax_p = ax_g.twinx()  # right y-axis for snapback pressure
        snap_line, = ax_p.plot([], [], lw=1.4, color="red", alpha=snap_alpha)
        snap_center_line, = ax_p.plot([], [], lw=1.0, color="0.4", alpha=0.7)
        snap_neg_fill = None
        ax_p.tick_params(axis="y", labelcolor="red")
        ax_p.set_ylabel("Snap", color="red")

    # State
    xs: list[float] = []
    ys: list[float] = []
    stall_x: list[float] = []
    stall_y: list[float] = []
    thrust_x: list[float] = []
    thrust_y: list[float] = []

    # Zoom state
    x_zoom_start: int | None = None
    y_zoom_start: int | None = None
    x_start_lim: tuple[float, float] | None = None
    y_start_lim: tuple[float, float] | None = None

    # --- interactivity / scoring state ---
    speed_ms_cur = int(speed_ms)   # live interval we can tweak
    paused = False

    holdings: list[float] = []     # entry prices
    score = 0.0                    # realized PnL from sells
    buys = 0
    sells = 0

    state = {
        "price": float(c[start_idx]),
        "ts": float(t[start_idx]),
    }  # latest seen price for key handler

    # Change total_frames so we start at a full window
    total_frames = N - window_size

    def update(frame):
        idx = frame + window_size  # shift so first frame shows full window
        left_idx = idx - window_size

        x_window = t[left_idx:idx+1]
        y_window = c[left_idx:idx+1]

        # Plot main line and overlays
        line.set_data(x_window, y_window)
        dot.set_data([x_window[-1]], [y_window[-1]])
        center_line.set_data(x_window, ema_k[left_idx:idx+1])
        upper_band.set_data(x_window, ema_k[left_idx:idx+1] + atr[left_idx:idx+1])
        lower_band.set_data(x_window, ema_k[left_idx:idx+1] - atr[left_idx:idx+1])

        # last sample in window is the current point we "see"
        state["price"] = float(y_window[-1])
        state["ts"] = float(x_window[-1])

        current_price = c[idx]
        marker_height = (all_time_high - all_time_low) * 0.01
        bar_marker.set_y(current_price - marker_height / 2)

        refresh_hud()

        # Keep window focus
        ax.set_xlim(float(x_window[0]), float(x_window[-1]))
        ax.set_ylim(*_pad((float(y_window.min()), float(y_window.max())), 0.02))
        ax_g.set_xlim(float(x_window[0]), float(x_window[-1]))

        if snap_on:
            snap_win = snap[left_idx:idx+1]
            snap_avg_win = snap_avg[left_idx:idx+1]
            snap_center_win = snap_center[left_idx:idx+1]

            snap_line.set_data(x_window, snap_win)
            blue_line.set_data(x_window, snap_avg_win)
            snap_center_line.set_data(x_window, snap_center_win)

            ymax = float(np.nanmax(snap_win)) if snap_win.size else 1.0
            ymin = float(np.nanmin(snap_win)) if snap_win.size else 0.0
            pad = max(0.2, 0.1 * (ymax - ymin if ymax > ymin else 1.0))
            ax_p.set_xlim(float(x_window[0]), float(x_window[-1]))
            ax_p.set_ylim(ymin - pad, ymax + pad)

            nonlocal snap_neg_fill
            if snap_neg_fill:
                snap_neg_fill.remove()
                snap_neg_fill = None
            if snap_fill and snap_win.size:
                snap_neg_fill = ax_p.fill_between(
                    x_window, snap_win, snap_center_win,
                    where=(snap_win < snap_center_win),
                    interpolate=True, color="red", alpha=0.12
                )

        objs = [
            line,
            dot,
            center_line,
            upper_band,
            lower_band,
            blue_line,
            title_right,
            status_text,
            bar_marker,
            bar_bg,
        ]
        if snap_on:
            objs.extend([snap_line, snap_center_line])
            if snap_neg_fill:
                objs.append(snap_neg_fill)
        return tuple(objs)

    anim = FuncAnimation(
        fig,
        update,
        frames=range(total_frames),
        interval=speed_ms_cur,
        blit=False,
        repeat=False,
    )

    def on_key(event):
        nonlocal paused, speed_ms_cur, score, buys, sells, holdings

        k = (event.key or "").lower()

        if k == "escape":
            plt.close(fig)
            return

        # pause / resume
        if k == " ":
            if paused:
                anim.event_source.start()
            else:
                anim.event_source.stop()
            paused = not paused
            return

        # ----- speed controls -----
        SPEED_UP = {"+", "=", "plus", "equal", "kp_add", "add"}
        SPEED_DOWN = {"-", "_", "minus", "kp_subtract", "subtract"}

        if k in SPEED_UP:
            speed_ms_cur = max(1, int(speed_ms_cur / 1.25))
            # reapply interval; stop/start to ensure backend picks it up
            was_running = not paused
            anim.event_source.stop()
            anim.event_source.interval = speed_ms_cur
            if was_running:
                anim.event_source.start()
            refresh_hud()
            status_text.set_text(f"speed ↑  {speed_ms_cur} ms")
            return

        if k in SPEED_DOWN:
            speed_ms_cur = min(2000, int(speed_ms_cur * 1.25))
            was_running = not paused
            anim.event_source.stop()
            anim.event_source.interval = speed_ms_cur
            if was_running:
                anim.event_source.start()
            refresh_hud()
            status_text.set_text(f"speed ↓  {speed_ms_cur} ms")
            return

        # ----- trading -----
        if k in {"x", "X"}:  # BUY 1
            holdings.append(state["price"])
            buys += 1
            refresh_hud()
            status_text.set_text(f"BUY @ {state['price']:.4f}")
            status_text.set_color("tab:green")
            return

        if k in {"z", "Z"}:  # SELL 1 (highest entry first)
            if holdings:
                i_max = max(range(len(holdings)), key=lambda i: holdings[i])
                entry = holdings.pop(i_max)
                pnl = state["price"] - entry
                score += pnl
                sells += 1
                refresh_hud()
                status_text.set_text(
                    f"SELL @ {state['price']:.4f}  PnL {pnl:+.4f}"
                )
                status_text.set_color("tab:red")
            else:
                status_text.set_text("no holdings to sell")
                status_text.set_color("0.5")
            return

    fig.canvas.mpl_connect("key_press_event", on_key)

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
