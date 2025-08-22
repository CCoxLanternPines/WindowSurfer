from __future__ import annotations
from typing import Dict, Any
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from .base import Brain, BrainOutput

def slope(x: np.ndarray) -> float:
    n = len(x)
    if n < 2:
        return 0.0
    xs = np.arange(n)
    xm, ym = xs.mean(), x.mean()
    num = ((xs - xm) * (x - ym)).sum()
    den = ((xs - xm)**2).sum() or 1.0
    return float(num / den)

class ExhaustionBrain(Brain):
    name = "exhaustion"
    settings_key = "strategy_settings"

    def compute(self, df: pd.DataFrame, settings: Dict[str, Any]) -> BrainOutput:
        st = settings["general_settings"][self.settings_key]
        max_pressure = int(st.get("max_pressure", 12))
        flat_deg = float(st.get("flat_band_deg", 8.0))
        k = int(st.get("window_size", 24))

        close = df["close"].to_numpy(float)
        n = len(close)

        # rolling slope for context
        k_eff = max(4, min(k, n))
        roll_slope = np.zeros(n)
        for t in range(n):
            a = max(0, t - k_eff + 1)
            roll_slope[t] = slope(close[a:t+1])

        # pressure runs (up/down); bubble size equals pressure length
        run_len = np.zeros(n, dtype=int)
        run_dir = np.zeros(n, dtype=int)  # +1 up, -1 down, 0 flat
        exh_flag = np.zeros(n, dtype=bool)
        exh_type = np.full(n, "", dtype=object)  # "sell"/"buy"/""
        bubble = np.zeros(n)

        cur_dir, cur_len = 0, 0
        for t in range(1, n):
            d = np.sign(close[t] - close[t-1])
            d = 0 if d == 0 else int(d)
            # continue run?
            if d == cur_dir and d != 0:
                cur_len += 1
            else:
                # run ended at t-1 => possible exhaustion
                if cur_dir != 0 and cur_len >= max_pressure:
                    exh_flag[t-1] = True
                    exh_type[t-1] = "sell" if cur_dir == +1 else "buy"
                    bubble[t-1] = float(cur_len)
                # start new
                cur_dir = d
                cur_len = 1 if d != 0 else 0
            run_len[t] = cur_len
            run_dir[t] = cur_dir

        # flat band mask (optional context)
        flat_mask = np.abs(np.rad2deg(np.arctan(roll_slope))) <= flat_deg

        feats = dict(
            roll_slope=roll_slope,
            run_len=run_len,
            run_dir=run_dir,           # +1/-1/0
            exh_flag=exh_flag,
            exh_type=exh_type,         # "sell"/"buy"/""
            bubble_size=bubble,
            flat_mask=flat_mask,
        )
        return BrainOutput(features=feats)

    def visualize(self, df: pd.DataFrame, out: BrainOutput, ax: plt.Axes) -> None:
        f = out.features
        idx = np.arange(len(df))
        ax.plot(idx, df["close"].to_numpy(float), color="blue", lw=1, label="Close")

        # red = up-run exhaustion; green = down-run exhaustion
        r = np.where((f["exh_flag"]) & (f["exh_type"] == "sell"))[0]
        g = np.where((f["exh_flag"]) & (f["exh_type"] == "buy"))[0]
        ax.scatter(r, df["close"].iloc[r], s=f["bubble_size"][r]*10, c="red", alpha=0.8)
        ax.scatter(g, df["close"].iloc[g], s=f["bubble_size"][g]*10, c="green", alpha=0.8)
        ax.legend(loc="upper left")

    def truth(self) -> Dict[str, Any]:
        def stats_uptrend_duration(df, f):
            # SELL exhaustion ends an up-run
            lens = f["run_len"][ (f["exh_flag"]) & (f["exh_type"] == "sell") ]
            sel = lens[lens > 50]
            return {"avg": float(sel.mean()) if len(sel) else None,
                    "median": float(np.median(sel)) if len(sel) else None,
                    "N": int(len(sel))}
        def stats_downtrend_duration(df, f):
            lens = f["run_len"][ (f["exh_flag"]) & (f["exh_type"] == "buy") ]
            sel = lens[lens > 50]
            return {"avg": float(sel.mean()) if len(sel) else None,
                    "median": float(np.median(sel)) if len(sel) else None,
                    "N": int(len(sel))}
        def slope_delta(df, f, kind: str):
            close = df["close"].to_numpy(float)
            T = np.where((f["exh_flag"]) & (f["exh_type"] == kind))[0]
            vals = []
            for t in T:
                if t-64 >= 0 and t+64 < len(close):
                    pre = slope(close[t-64:t])
                    post = slope(close[t:t+64])
                    vals.append(post - pre)
            vals = np.array(vals)
            return {"avg": float(vals.mean()) if len(vals) else None,
                    "median": float(np.median(vals)) if len(vals) else None,
                    "N": int(len(vals))}
        def bubble_stats(df, f, kind: str):
            b = f["bubble_size"][ (f["exh_flag"]) & (f["exh_type"] == kind) ]
            return {"avg": float(b.mean()) if len(b) else None,
                    "median": float(np.median(b)) if len(b) else None,
                    "N": int(len(b))}
        return {
            "Uptrend duration >50 (SELL exhs)": lambda df, fo: stats_uptrend_duration(df, fo),
            "Downtrend duration >50 (BUY exhs)": lambda df, fo: stats_downtrend_duration(df, fo),
            "Slope Δ (SELL exhs, 64)":         lambda df, fo: slope_delta(df, fo, "sell"),
            "Slope Δ (BUY exhs, 64)":          lambda df, fo: slope_delta(df, fo, "buy"),
            "Bubble size (SELL exhs)":         lambda df, fo: bubble_stats(df, fo, "sell"),
            "Bubble size (BUY exhs)":          lambda df, fo: bubble_stats(df, fo, "buy"),
        }
