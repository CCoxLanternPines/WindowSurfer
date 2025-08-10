from __future__ import annotations

import argparse

from systems import brain_score


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["test"], required=True)
    parser.add_argument("--brain", choices=["bear", "chop", "bull"], required=True)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--out")
    parser.add_argument("-v", dest="verbose", action="count", default=0)
    args = parser.parse_args()

    if args.mode == "test":
        brain_score.run_single_brain(args)


if __name__ == "__main__":
    main()

