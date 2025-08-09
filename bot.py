"""CLI entry for Phase 0 simulator."""

from __future__ import annotations

import argparse
import json

from systems.sim_engine import run_sim


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", required=True)
    parser.add_argument("--trials", type=int, default=10)
    args = parser.parse_args()

    mode = args.mode.lower()
    if mode == "sim":
        with open("settings/settings.json") as f:
            settings = json.load(f)
        run_sim(settings)
    elif mode == "tune":
        from optimizer import run as run_optimizer

        run_optimizer(args.trials)
    else:
        parser.error("--mode must be either 'sim' or 'tune'")


if __name__ == "__main__":
    main()
