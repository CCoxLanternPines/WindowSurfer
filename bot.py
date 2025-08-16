from __future__ import annotations

"""Minimal command-line entry point for discovery simulation."""

import argparse

from systems import sim_engine


def main() -> None:
    parser = argparse.ArgumentParser(description="WindowSurfer discovery bot")
    parser.add_argument("--mode", required=True, help="Mode to run (sim)")
    parser.add_argument("--time", default="1m", help="Time window for simulation")
    parser.add_argument(
        "--pressure-lookback",
        type=int,
        default=sim_engine.PRESSURE_LOOKBACK,
        help="Lookback candles for pressure bias",
    )
    parser.add_argument(
        "--pressure-scale",
        type=float,
        default=sim_engine.PRESSURE_SCALE,
        help="Scaling factor for pressure bias",
    )
    parser.add_argument(
        "--no-pressure-bias",
        dest="enable_pressure_bias",
        action="store_false",
        default=True,
        help="Disable pressure bias",
    )
    parser.add_argument(
        "--control-steps",
        dest="enable_control_steps",
        action="store_true",
        default=sim_engine.ENABLE_CONTROL_STEPS,
        help="Use stepped control line instead of continuous",
    )
    parser.add_argument(
        "--debug-plots", action="store_true", help="Show pressure bias overlay"
    )
    parser.add_argument(
        "-v", "--verbose", action="count", default=0, help="Increase verbosity"
    )
    args = parser.parse_args()

    if args.mode.lower() == "sim":
        sim_engine.run_simulation(
            timeframe=args.time,
            verbose=args.verbose,
            debug_plots=args.debug_plots,
            enable_pressure_bias=args.enable_pressure_bias,
            enable_control_steps=args.enable_control_steps,
            pressure_lookback=args.pressure_lookback,
            pressure_scale=args.pressure_scale,
        )
    else:
        raise ValueError(f"Unknown mode: {args.mode}")


if __name__ == "__main__":
    main()
