from __future__ import annotations

"""Minimal command-line entry point for discovery simulation."""

import argparse

from systems import sim_engine


def main() -> None:
    parser = argparse.ArgumentParser(description="WindowSurfer discovery bot")
    parser.add_argument("--mode", required=True, help="Mode to run (sim)")
    parser.add_argument(
        "--time",
        default="1m",
        help="Time window for simulation (e.g. 1m for one month)",
    )
    args = parser.parse_args()

    if args.mode.lower() == "sim":
        sim_engine.run_simulation(timeframe=args.time)
    else:
        raise ValueError(f"Unknown mode: {args.mode}")


if __name__ == "__main__":
    main()
