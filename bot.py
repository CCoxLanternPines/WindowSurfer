from __future__ import annotations

import argparse

from systems import brain_score


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd")

    brains_p = sub.add_parser("brains")
    brains_sub = brains_p.add_subparsers(dest="action")
    score_p = brains_sub.add_parser("score")
    score_p.add_argument("--coins", required=True)
    score_p.add_argument("--start", required=True)
    score_p.add_argument("--end", required=True)
    score_p.add_argument("--horizons")
    score_p.add_argument("--out")
    score_p.add_argument("--min_events", type=int)
    score_p.add_argument("-v", dest="verbose", action="count", default=0)

    args = parser.parse_args()

    if args.cmd == "brains" and args.action == "score":
        brain_score.main(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
