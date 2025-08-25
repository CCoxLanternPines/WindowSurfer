#!/usr/bin/env python3
"""Thin CLI wrapper for simulation visualization."""

from __future__ import annotations

import argparse
from typing import Optional

from systems.graph_engine import render_simulation


def main(argv: Optional[list[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Render simulation ledger")
    parser.add_argument(
        "--ledger",
        default="data/temp/sim_data.json",
        help="Path to simulation ledger JSON",
    )
    parser.add_argument(
        "--viz", action="store_true", help="Render visualization"
    )
    args = parser.parse_args(argv)

    if args.viz:
        render_simulation(args.ledger)


if __name__ == "__main__":  # pragma: no cover
    main()

