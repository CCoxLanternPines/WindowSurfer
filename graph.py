#!/usr/bin/env python3
"""CLI wrapper for rendering graph feeds."""

from __future__ import annotations

import argparse
from typing import Optional

from pathlib import Path

from systems.graph_engine import discover_feed, render_feed


def main(argv: Optional[list[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Render graph feed")
    parser.add_argument("--feed", help="Explicit feed path", default=None)
    parser.add_argument("--mode", choices=["sim", "live"], default="sim")
    parser.add_argument("--coin", required=False, default="")
    parser.add_argument("--account", default=None)
    parser.add_argument("--follow", action="store_true", help="Tail feed updates")
    parser.add_argument("--viz", action="store_true", help="Render visualization")
    args = parser.parse_args(argv)

    if not args.feed:
        if not args.coin:
            parser.error("--coin required when --feed not specified")
        path = discover_feed(
            mode=args.mode,
            coin=args.coin,
            account=args.account,
        )
    else:
        path = Path(args.feed)

    if args.viz:
        render_feed(path, follow=args.follow)


if __name__ == "__main__":  # pragma: no cover
    main()

