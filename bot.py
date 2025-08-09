"""CLI entry for Phase 0 simulator."""

from __future__ import annotations

import argparse
import json

from systems.sim_engine import run_sim


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", required=True)
    args = parser.parse_args()
    if args.mode.lower() != "sim":
        parser.error("--mode sim is the only supported mode")
    with open("settings/settings.json") as f:
        settings = json.load(f)
    run_sim(settings)


if __name__ == "__main__":
    main()
