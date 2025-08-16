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
DEFAULT_BOTTOM_WINDOW = "24h"

# Forecast weights
WEIGHT_PERSISTENCE = 0.5
WEIGHT_ERROR = 0.2
WEIGHT_VOLUME = 0.2
WEIGHT_VOLATILITY = 0.08

# Forecast confidence threshold
CONFIDENCE_THRESHOLD = 0.1  # only use forecasts above this

# Control line thresholds
ENTRY_THRESHOLD = 0.2
EXIT_THRESHOLD = 0.03

# Plotting constants
CONTROL_PANEL_HEIGHTS = (4, 1)  # matplotlib height ratios for (price, control)


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


def parse_window_size(window: str | int, base_candle: str = "1h") -> int:
    """
    Convert a string like '2d', '1w', '1m' into number of candles,
    assuming each candle = 1h by default.
    If already an int, just return it.
    """
    if isinstance(window, int):
        return window
    if isinstance(window, float):  # legacy % values
        return max(1, int(len(df) * window))  # careful: df must exist here
    if isinstance(window, str):
        delta = parse_timeframe(window)
        if not delta:
            raise ValueError(f"Could not parse window size: {window}")
        if base_candle == "1h":
            return int(delta.total_seconds() // 3600)
        if base_candle == "1d":
            return int(delta.total_seconds() // (3600 * 24))
    raise TypeError(f"Unsupported window type: {type(window)}")

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

    BOTTOM_WINDOW = parse_window_size(DEFAULT_BOTTOM_WINDOW)

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
    forecast_confidences = [np.nan] * len(df)

    # track persistence (consecutive slope runs)
    last_slope_sign = 0
    persistence_count = 0

    # accuracy tracking
    total_forecasts = 0
    total_correct = 0
    total_weighted = 0.0
    total_conf = 0.0

    total_filtered = 0
    correct_filtered = 0
    weighted_filtered = 0.0
    conf_filtered = 0.0

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

            # Confidence = absolute value of forecast angle
            confidence = abs(forecast_angle)

            # actual slope sign
            actual_sign = np.sign(actual_angle)
            pred_sign = np.sign(forecast_angle)

            # --- raw accuracy ---
            total_forecasts += 1
            if pred_sign == actual_sign:
                total_correct += 1
            total_weighted += (1 if pred_sign == actual_sign else 0) * confidence
            total_conf += confidence

            # --- filtered accuracy ---
            if confidence >= CONFIDENCE_THRESHOLD:
                total_filtered += 1
                if pred_sign == actual_sign:
                    correct_filtered += 1
                weighted_filtered += (
                    (1 if pred_sign == actual_sign else 0) * confidence
                )
                conf_filtered += confidence

            # assign to full window
            forecast_angles[i:end] = [forecast_angle] * (end - i)
            forecast_confidences[i:end] = [confidence] * (end - i)

    df["forecast_angle"] = forecast_angles
    df["confidence"] = forecast_confidences

    # --- Control Line Generation ---
    control_line: list[float] = []
    signal_counts: Dict[float, int] = {}
    for slope, conf in zip(df["forecast_angle"], df["confidence"]):
        if slope >= 0 and conf >= ENTRY_THRESHOLD * 2:
            val = 1.0
        elif slope >= 0 and conf >= ENTRY_THRESHOLD:
            val = 0.5
        elif slope <= 0 and conf >= EXIT_THRESHOLD * 2:
            val = -1.0
        elif slope <= 0 and conf >= EXIT_THRESHOLD:
            val = -0.5
        else:
            val = 0.0
        control_line.append(val)
        signal_counts[val] = signal_counts.get(val, 0) + 1

    df["control_line"] = control_line

    total_signals = len([s for s in control_line if s != 0])
    correct_signals = 0
    weighted_correct = 0.0
    total_signal_conf = 0.0
    for idx, signal in enumerate(control_line):
        if signal == 0:
            continue
        actual_sign = np.sign(df["slope_angle"].iloc[idx])
        conf = df["confidence"].iloc[idx]
        if np.sign(signal) == actual_sign:
            correct_signals += 1
            weighted_correct += conf
        total_signal_conf += conf

    raw_signal_acc = (
        (correct_signals / total_signals * 100) if total_signals else 0
    )
    weighted_signal_acc = (
        (weighted_correct / total_signal_conf * 100) if total_signal_conf else 0
    )

    raw_acc = (total_correct / total_forecasts * 100) if total_forecasts else 0
    raw_weighted = (total_weighted / total_conf * 100) if total_conf else 0
    filt_acc = (correct_filtered / total_filtered * 100) if total_filtered else 0
    filt_weighted = (weighted_filtered / conf_filtered * 100) if conf_filtered else 0

    print(f"[SIM] Raw Accuracy: {raw_acc:.2f}% | Weighted: {raw_weighted:.2f}%")
    print(
        f"[SIM] Filtered Accuracy (conf ≥ {CONFIDENCE_THRESHOLD}): {filt_acc:.2f}% | Weighted: {filt_weighted:.2f}%"
    )
    print(
        f"[SIM] Forecasts: {total_forecasts}, Used: {total_filtered}, Skipped: {total_forecasts - total_filtered}"
    )

    print(
        f"[SIM] Control Line Accuracy: {raw_signal_acc:.2f}% | Weighted: {weighted_signal_acc:.2f}%"
    )
    print(
        f"[SIM] Signal Counts -> +1: {signal_counts.get(1.0, 0)}, +0.5: {signal_counts.get(0.5, 0)}, 0: {signal_counts.get(0.0, 0)}, -0.5: {signal_counts.get(-0.5, 0)}, -1: {signal_counts.get(-1.0, 0)}"
    )

    state: Dict[str, Any] = {}
    buy_points = []
    for _, candle in df.iterrows():
        before_state = dict(state)
        evaluate_buy.evaluate_buy(candle.to_dict(), state)
        if state.get("last_action") == "buy":
            buy_points.append((candle["candle_index"], candle["close"]))
        evaluate_sell.evaluate_sell(candle.to_dict(), state)

    fig, (ax1, ax2) = plt.subplots(
        2,
        1,
        sharex=True,
        figsize=(12, 6),
        gridspec_kw={"height_ratios": CONTROL_PANEL_HEIGHTS},
    )
    ax1.plot(df["candle_index"], df["close"], label="Close Price", color="blue")
    ax1.plot(
        df["candle_index"],
        df["bottom_slope"],
        label=f"Slope Line ({BOTTOM_WINDOW})",
        color="black",
        linewidth=2,
        drawstyle="steps-post",
    )
    if buy_points:
        bx, by = zip(*buy_points)
        ax1.scatter(
            bx,
            by,
            s=80,
            color="green",
            edgecolor="black",
            zorder=5,
            marker="o",
            label=f"Buys ({len(buy_points)})",
        )
    ax1.set_ylabel("Price")
    ax1.legend(loc="upper left")

    ax2.step(
        df["candle_index"], df["control_line"], where="mid", color="red"
    )
    ax2.set_ylim(-1.2, 1.2)
    ax2.set_yticks([-1, -0.5, 0, 0.5, 1])
    ax2.set_yticklabels(
        [
            "Dump Hard (-1)",
            "Dump Light (-0.5)",
            "Neutral (0)",
            "Hold Light (+0.5)",
            "Hold Hard (+1)",
        ]
    )
    ax2.set_xlabel("Candles (Index)")
    ax2.set_title("Control Line (Exit Oracle)")

    fig.suptitle("SOLUSD Discovery Simulation")
    ax1.grid(True)
    ax2.grid(True)
    plt.show()
