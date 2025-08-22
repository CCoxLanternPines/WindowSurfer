from __future__ import annotations

"""Minimal command-line entry point for discovery simulation."""

from systems.utils.cli import build_parser
from systems import sim_engine


def main() -> None:
    parser = build_parser()
    parser.add_argument("--time", default="1m", help="Time window for simulation")
    args = parser.parse_args()

    if args.mode and args.mode.lower() == "sim":
        sim_engine.run_simulation(timeframe=args.time, brain=args.brain)
    else:
        raise ValueError(f"Unknown mode: {args.mode}")


if __name__ == "__main__":
    main()
