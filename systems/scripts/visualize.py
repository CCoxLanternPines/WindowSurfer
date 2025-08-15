from __future__ import annotations

def plot_trades(candles_df, events, out_path="data/tmp/sim_plot.png", show=True) -> None:
    # Lazy import to keep non-viz runs fast
    import matplotlib.pyplot as plt

    # Basic line of close vs time
    x = candles_df["time"]
    y = candles_df["close"]
    fig, ax = plt.subplots()
    ax.plot(x, y, linewidth=1)

    # Scatter buys/sells
    buys  = [(e["time"], e["price"]) for e in events if e["type"] == "buy"]
    sells = [(e["time"], e["price"]) for e in events if e["type"] == "sell"]
    if buys:
        bx, by = zip(*buys)
        ax.scatter(bx, by, marker="^", s=36, alpha=0.7, label=f"Buys ({len(buys)})")
    if sells:
        sx, sy = zip(*sells)
        ax.scatter(sx, sy, marker="v", s=36, alpha=0.7, label=f"Sells ({len(sells)})")

    if not events:
        ax.text(0.02, 0.95, "No trades", transform=ax.transAxes)

    ax.set_title("Simulation — Price with Buy/Sell Markers")
    ax.set_xlabel("Time")
    ax.set_ylabel("Price")
    ax.legend(loc="best")

    # Always save; best-effort show
    import os
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    try:
        if show:
            plt.show()
    except Exception:
        pass
    finally:
        plt.close(fig)
