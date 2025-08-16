from __future__ import annotations

"""Very small historical simulation engine."""

import re
from datetime import timedelta
from typing import Any, Dict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .scripts import evaluate_buy, evaluate_sell


# Step size (candles) for slope updates
# If < 1, treated as fraction of dataset length
DEFAULT_BOTTOM_WINDOW = 0.1

# Forecast weights
WEIGHT_PERSISTENCE = 0.5
WEIGHT_ERROR = 0.2
WEIGHT_VOLUME = 0.2
WEIGHT_VOLATILITY = 0.1


def parse_timeframe(tf: str) -> timedelta | None:
    match = re.match(r"(\d+)([dhmw])", tf)
    if not match:
        return None
    val, unit = int(match.group(1)), match.group(2)
    if unit == "d":
        return timedelta(days=val)
    if unit == "w":
        return timedelta(weeks=val)
    if unit == "m":
        return timedelta(days=30 * val)  # rough month
    if unit == "h":
        return timedelta(hours=val)
    return None


def run_simulation(*, timeframe: str = "1m") -> None:
    """Run a simple simulation over SOLUSD candles."""
    file_path = "data/sim/SOLUSD_1h.csv"
    df = pd.read_csv(file_path)

    if timeframe:
        delta = parse_timeframe(timeframe)
        if delta:
            cutoff = (pd.Timestamp.utcnow().tz_localize(None) - delta).timestamp()
            df = df[df["timestamp"] >= cutoff]

    df = df.reset_index(drop=True)
    df["candle_index"] = range(len(df))

    total_candles = len(df)
    if DEFAULT_BOTTOM_WINDOW < 1:
        BOTTOM_WINDOW = max(1, int(total_candles * DEFAULT_BOTTOM_WINDOW))
    else:
        BOTTOM_WINDOW = int(DEFAULT_BOTTOM_WINDOW)

    print(
        f"[SIM] Using BOTTOM_WINDOW={BOTTOM_WINDOW} (derived from {DEFAULT_BOTTOM_WINDOW})"
    )

    # Stepwise slope calculation
    slopes = [np.nan] * len(df)
    slope_angles = [np.nan] * len(df)
    last_value = df["close"].iloc[0]

    for i in range(0, len(df), BOTTOM_WINDOW):
        end = min(i + BOTTOM_WINDOW, len(df))
        y = df["close"].iloc[i:end].values
        x = np.arange(len(y))
        if len(y) > 1:
            m, b = np.polyfit(x, y, 1)
            fitted = last_value + m * np.arange(len(y))
            slopes[i:end] = fitted
            last_value = fitted[-1]
            slope_val = np.tanh(m)
            slope_angles[i:end] = [slope_val] * len(y)
        else:
            slopes[i:end] = [last_value] * len(y)
            slope_angles[i:end] = [0] * len(y)

    df["bottom_slope"] = slopes
    df["slope_angle"] = slope_angles

    # Forward slope forecasting
    forecast_angles = [np.nan] * len(df)

    # track persistence (consecutive slope runs)
    last_slope_sign = 0
    persistence_count = 0

    for i in range(0, len(df), BOTTOM_WINDOW):
        end = min(i + BOTTOM_WINDOW, len(df))
        y = df["close"].iloc[i:end].values
        x = np.arange(len(y))

        if len(y) > 1:
            m, b = np.polyfit(x, y, 1)
            actual_angle = np.tanh(m)  # normalized slope angle

            # --- Baseline (Codex 1): persistence ---
            slope_sign = np.sign(actual_angle)
            if slope_sign == last_slope_sign:
                persistence_count += 1
            else:
                persistence_count = 1
            last_slope_sign = slope_sign
            score_persistence = slope_sign * (persistence_count / 5.0)  # scaled

            # --- Improved (Codex 2): error + volume bias ---
            # residual error (fit quality)
            y_fit = m * x + b
            residuals = y - y_fit
            fit_error = np.mean(np.abs(residuals))
            score_error = -np.sign(actual_angle) * (fit_error / np.std(y))

            # volume bias
            recent_vol = df["volume"].iloc[i:end]
            vol_change = (recent_vol.iloc[-1] - recent_vol.iloc[0]) / max(
                1e-9, recent_vol.iloc[0]
            )
            score_volume = np.sign(vol_change) * abs(vol_change)

            # --- Target (Codex 3): volatility + multi-window alignment ---
            # volatility (ATR proxy)
            atr = (df["high"].iloc[i:end] - df["low"].iloc[i:end]).mean()
            score_volatility = -np.sign(actual_angle) * (
                atr / max(1e-9, np.mean(df["close"].iloc[i:end]))
            )

            # multi-window alignment: compare current slope vs. longer context
            context_size = min(len(df), BOTTOM_WINDOW * 3)
            ctx_y = df["close"].iloc[max(0, i - context_size) : end].values
            ctx_x = np.arange(len(ctx_y))
            if len(ctx_y) > 1:
                m_ctx, _ = np.polyfit(ctx_x, ctx_y, 1)
                ctx_angle = np.tanh(m_ctx)
            else:
                ctx_angle = 0
            score_context = np.sign(ctx_angle) * abs(ctx_angle)

            # --- Combine all features ---
            forecast_angle = (
                WEIGHT_PERSISTENCE * score_persistence
                + WEIGHT_ERROR * score_error
                + WEIGHT_VOLUME * score_volume
                + WEIGHT_VOLATILITY * score_volatility
                + (1 - (
                    WEIGHT_PERSISTENCE
                    + WEIGHT_ERROR
                    + WEIGHT_VOLUME
                    + WEIGHT_VOLATILITY
                ))
                * score_context
            )

            # clamp [-1,1]
            forecast_angle = max(-1, min(1, forecast_angle))

            # assign to full window
            forecast_angles[i:end] = [forecast_angle] * (end - i)

    df["forecast_angle"] = forecast_angles

    state: Dict[str, Any] = {}
    for _, candle in df.iterrows():
        evaluate_buy.evaluate_buy(candle.to_dict(), state)
        evaluate_sell.evaluate_sell(candle.to_dict(), state)

    fig, ax1 = plt.subplots(figsize=(12, 6))
    ax1.plot(df["candle_index"], df["close"], label="Close Price", color="blue")
    ax1.plot(
        df["candle_index"],
        df["bottom_slope"],
        label=f"Slope Line ({BOTTOM_WINDOW})",
        color="black",
        linewidth=2,
        drawstyle="steps-post",
    )
    ax1.set_ylabel("Price")
    ax1.set_xlabel("Candles (Index)")
    ax1.legend(loc="upper left")

    ax2 = ax1.twinx()
    ax2.plot(
        df["candle_index"],
        df["forecast_angle"],
        label="Forecast Slope Angle",
        color="red",
        drawstyle="steps-post",
    )
    ax2.set_ylim(-1, 1)
    ax2.axhline(0, color="gray", linestyle="--", linewidth=1)
    ax2.set_ylabel("Slope Angle [-1,1]")
    ax2.legend(loc="lower right")

    plt.title("SOLUSD Discovery Simulation")
    plt.grid(True)
    plt.show()
