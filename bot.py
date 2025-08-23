from __future__ import annotations

"""Minimal command-line entry point for discovery simulation."""

# This entry script routes simulation runs through the MetaBrain arbiter
# which now supports regime-aware weighted scoring.

import argparse

from systems import sim_engine, brain_engine


def main() -> None:
    parser = argparse.ArgumentParser(description="WindowSurfer discovery bot")
    parser.add_argument("--mode", required=True, help="Mode to run (sim, brain, live)")
    parser.add_argument(
        "--time",
        default="1m",
        help="Time window for simulation (e.g., 1m for one month)",
    )
    brains = brain_engine.list_brains()
    parser.add_argument(
        "--brain",
        choices=brains,
        help="Name of brain module to run",
    )
    parser.add_argument("--viz", action="store_true", help="Show plot for the run")
    args = parser.parse_args()

    if args.mode.lower() == "sim":
        sim_engine.run_simulation(timeframe=args.time, viz=args.viz)
    elif args.mode.lower() == "brain":
        if not args.brain:
            print("Available brains:")
            for name in brains:
                print(f"- {name}")
            return
        brain_engine.run_brain(args.brain, args.time, args.viz)
    elif args.mode.lower() == "live":
        from systems.live_engine import run_live

        run_live(timeframe=args.time)
    else:
        raise ValueError(f"Unknown mode: {args.mode}")


if __name__ == "__main__":
    main()
