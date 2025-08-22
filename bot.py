from __future__ import annotations

"""Minimal command-line entry point for discovery simulation."""

import argparse

from systems import sim_engine, brain_runner
from systems.brains import list_brains


def main() -> None:
    parser = argparse.ArgumentParser(description="WindowSurfer discovery bot")
    parser.add_argument("--mode", required=True, help="Mode to run (sim or brain)")
    parser.add_argument(
        "--time",
        default="1m",
        help="Time window (e.g. '3m' for three months)",
    )
    parser.add_argument("--brain", help="Brain to run in --mode brain")
    args = parser.parse_args()

    mode = args.mode.lower()
    if mode == "sim":
        sim_engine.run_simulation(timeframe=args.time)
    elif mode == "brain":
        if not args.brain:
            print("Available brains:")
            for name in list_brains():
                print("-", name)
            return
        brain_runner.run(args.brain, timeframe=args.time, viz=True)
    else:
        raise ValueError(f"Unknown mode: {args.mode}")


if __name__ == "__main__":
    main()
