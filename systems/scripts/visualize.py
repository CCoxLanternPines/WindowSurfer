from __future__ import annotations

def plot_trades(
    candles_df,
    events,
    regime=None,
    out_path="data/tmp/sim_plot.png",
    show=True,
) -> None:
    # Lazy import to keep non-viz runs fast
    import matplotlib.pyplot as plt

    x = range(len(candles_df))
    y = candles_df["close"].values

    if regime:
        fig, (ax, ax_r) = plt.subplots(
            2, 1, sharex=True, gridspec_kw={"height_ratios": [3, 1]}
        )
    else:
        fig, ax = plt.subplots()
        ax_r = None
    ax.plot(x, y, linewidth=1)

    if regime:
        labels = regime.get("label", [])
        start = 0
        curr = labels[0] if labels else 0
        for i, lab in enumerate(labels + [None]):
            if lab != curr:
                color = {1: "green", 0: "gray", -1: "red"}.get(curr)
                if color is not None:
                    ax.axvspan(start, i, color=color, alpha=0.1)
                start = i
                curr = lab
        ax_r.plot(x, regime.get("score", []), linewidth=1)
        ax_r.axhline(0, color="black", linewidth=0.5)
        ax_r.set_ylim(-1, 1)
        ax_r.set_ylabel("Regime")

    # Map candle times to index for scatter buys/sells
    idx_map = {t: i for i, t in enumerate(candles_df["time"])}
    buys = [
        (idx_map[e["time"]], e["price"])
        for e in events
        if e["type"] == "buy" and e["time"] in idx_map
    ]
    sells = [
        (idx_map[e["time"]], e["price"])
        for e in events
        if e["type"] == "sell" and e["time"] in idx_map
    ]
    if buys:
        bx, by = zip(*buys)
        ax.scatter(bx, by, marker="^", s=36, alpha=0.7, label=f"Buys ({len(buys)})")
    if sells:
        sx, sy = zip(*sells)
        ax.scatter(sx, sy, marker="v", s=36, alpha=0.7, label=f"Sells ({len(sells)})")

    if not events:
        ax.text(0.02, 0.95, "No trades", transform=ax.transAxes)

    ax.set_title("Simulation — Price with Buy/Sell Markers")
    ax.set_xlabel("Candle #")
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
