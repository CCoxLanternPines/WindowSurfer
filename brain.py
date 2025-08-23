from __future__ import annotations

"""Dedicated CLI for auditing and teaching brains."""

import argparse

from systems import brain_engine, teach_engine


def main() -> None:
    parser = argparse.ArgumentParser(description="Brain workflow utilities")
    parser.add_argument("--brain", help="Name of brain module to run")
    parser.add_argument(
        "--time",
        default="1M",
        help="Lookback window for analysis (e.g., 1W, 3M)",
    )
    parser.add_argument("--viz", action="store_true", help="Enable visualization")

    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--audit", action="store_true", help="Run audit mode")
    modes.add_argument("--teach", action="store_true", help="Run teaching mode")
    modes.add_argument("--correct", action="store_true", help="Run correction mode")

    parser.add_argument("--list", action="store_true", help="List available brains")

    args = parser.parse_args()

    if args.list:
        brains = brain_engine.list_brains()
        for b in brains:
            print(b)
        return

    if not args.brain:
        parser.error("--brain is required unless --list is specified")

    if args.audit:
        print(f"[BRAIN][AUDIT][{args.time}] {args.brain}")
        teach_engine.run_audit(args.brain, args.time)
    elif args.teach:
        print(f"[BRAIN][TEACH][{args.time}] {args.brain}")
        teach_engine.run_teach(args.brain, args.time, args.viz)
    elif args.correct:
        print(f"[BRAIN][CORRECT][{args.time}] {args.brain}")
        teach_engine.run_correct(args.brain, args.time, args.viz)
    else:
        parser.error("choose one of --audit, --teach, or --correct")


if __name__ == "__main__":
    main()
