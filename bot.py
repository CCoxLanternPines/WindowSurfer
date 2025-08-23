from __future__ import annotations

"""Minimal command-line entry point for discovery and brain modes."""

import argparse
import sys

from systems import sim_engine


def addlog(message: str) -> None:
    """Simple logging helper mirroring legacy addlog."""
    print(message)


def main() -> None:
    parser = argparse.ArgumentParser(description="WindowSurfer discovery bot")
    parser.add_argument("--mode", required=True, help="Mode to run (sim or brain)")
    parser.add_argument(
        "--time",
        default="1m",
        help="Time window for simulation (e.g. 1m for one month)",
    )
    parser.add_argument("--ledger", help="Ledger name for config")
    parser.add_argument("--brain", help="Brain module name under systems.brains.*")
    parser.add_argument("--verbose", action="store_true", help="Verbose output")
    parser.add_argument("--viz", action="store_true", help="Enable visualization")
    args = parser.parse_args()

    mode = args.mode.lower()
    if mode == "sim":
        sim_engine.run_simulation(timeframe=args.time, viz=args.viz)
    elif mode == "brain":
        if not args.ledger:
            addlog("Error: --ledger is required for brain mode")
            sys.exit(1)
        if not args.brain:
            addlog("Error: --brain is required (e.g. exhaustion_demo)")
            sys.exit(1)
        from systems.brain_engine import run_brain

        run_brain(
            ledger=args.ledger,
            brain_name=args.brain,
            time_window=args.time,
            verbose=args.verbose,
            viz=args.viz,
        )
    else:
        raise ValueError(f"Unknown mode: {args.mode}")


if __name__ == "__main__":
    main()
