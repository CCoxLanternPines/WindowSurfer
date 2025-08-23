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
    parser.add_argument("--symbol", default="SOLUSD", help="Trading pair symbol")
    parser.add_argument("--chart", action="store_true", help="Show audit chart")
    parser.add_argument("--horizon", type=int, default=12, help="Outcome horizon in candles")
    parser.add_argument(
        "--use-labels",
        action="store_true",
        help="Use saved labels instead of proxy outcomes",
    )
    parser.add_argument(
        "--reaudit",
        action="store_true",
        help="After teaching, immediately audit using labels",
    )

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
        print("Hint: Use --symbol DOGEUSD --time 1Y")
        return

    if not args.brain:
        parser.error("--brain is required unless --list is specified")

    if args.audit:
        print(f"[BRAIN][AUDIT][{args.time}] {args.brain}")
        teach_engine.run_audit(
            args.brain,
            args.time,
            symbol=args.symbol,
            chart=args.chart,
            horizon=args.horizon,
            use_labels=args.use_labels,
        )
    elif args.teach:
        print(f"[BRAIN][TEACH][{args.time}] {args.brain}")
        teach_engine.run_teach(
            args.brain,
            args.time,
            args.viz,
            symbol=args.symbol,
            horizon=args.horizon,
            reaudit=args.reaudit,
        )
    elif args.correct:
        print(f"[BRAIN][CORRECT][{args.time}] {args.brain}")
        teach_engine.run_correct(
            args.brain, args.time, args.viz, symbol=args.symbol
        )
    else:
        parser.error("choose one of --audit, --teach, or --correct")


if __name__ == "__main__":
    main()
