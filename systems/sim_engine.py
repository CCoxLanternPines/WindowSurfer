def run_simulation(
    *,
    timeframe: str = "1m",
    verbose: int = 0,
    debug_plots: bool = False,
    enable_pressure_bias: bool = ENABLE_PRESSURE_BIAS,
    enable_control_steps: bool = ENABLE_CONTROL_STEPS,
    pressure_lookback: int = PRESSURE_LOOKBACK,
    pressure_scale: float = PRESSURE_SCALE,
) -> None:
    ...
    # --- Control Line Generation ---
    control_line: list[float] = []
    slope_signals: list[float] = []
    signal_counts: Dict[float, int] = {}

    if enable_control_steps:
        # stepped thresholds
        for idx, (slope, conf, bias, pos) in enumerate(
            zip(df["forecast_angle"], df["confidence"], bias_series, pos_series)
        ):
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

            slope_signals.append(val)
            signal_counts[val] = signal_counts.get(val, 0) + 1

            final_val = val + bias
            final_val = max(-1.0, min(1.0, final_val))
            control_line.append(final_val)

            if verbose >= 3 and idx % 100 == 0:
                print(f"[PRESSURE] pos={pos:.3f}, bias={bias:.3f}")

        # force last signal to exit (avoid dangling buys)
        if control_line and control_line[-1] > 0:
            last_slope = slope_signals[-1]
            signal_counts[last_slope] -= 1
            slope_signals[-1] = -1.0
            control_line[-1] = -1.0
            signal_counts[-1.0] = signal_counts.get(-1.0, 0) + 1

    else:
        # continuous candle-by-candle bias blending
        for idx, (slope, bias, pos) in enumerate(
            zip(df["forecast_angle"], bias_series, pos_series)
        ):
            val = slope + bias
            val = max(-1.0, min(1.0, val))
            control_line.append(val)

            if verbose >= 3 and idx % 100 == 0:
                print(f"[PRESSURE] pos={pos:.3f}, bias={bias:.3f}")

        if control_line and control_line[-1] > 0:
            control_line[-1] = -1.0

    df["control_line"] = control_line

    # --- Accuracy metrics ---
    if enable_control_steps:
        total_signals = len([s for s in slope_signals if s != 0])
        correct_signals = 0
        weighted_correct = 0.0
        total_signal_conf = 0.0
        for idx, (signal, slope_sig) in enumerate(zip(control_line, slope_signals)):
            if slope_sig == 0:
                continue
            actual_sign = np.sign(df["slope_angle"].iloc[idx])
            conf = df["confidence"].iloc[idx]
            total_signal_conf += conf
            if np.sign(signal) == np.sign(actual_sign):
                correct_signals += 1
                weighted_correct += conf
        raw_signal_acc = (correct_signals / total_signals * 100) if total_signals else 0
        weighted_signal_acc = (weighted_correct / total_signal_conf * 100) if total_signal_conf else 0
    else:
        dir_total = len(control_line)
        dir_correct = 0
        weighted_correct = 0.0
        weight_total = 0.0
        for idx, val in enumerate(control_line):
            actual = df["slope_angle"].iloc[idx]
            conf = df["confidence"].iloc[idx]
            weight = abs(actual) * conf
            if np.sign(val) == np.sign(actual):
                dir_correct += 1
                weighted_correct += weight
            weight_total += weight
        raw_signal_acc = (dir_correct / dir_total * 100) if dir_total else 0
        weighted_signal_acc = (weighted_correct / weight_total * 100) if weight_total else 0

    print(
        f"[SIM] Control Line Directional Accuracy: {raw_signal_acc:.2f}% | Weighted: {weighted_signal_acc:.2f}%"
    )
