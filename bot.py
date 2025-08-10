from __future__ import annotations

import argparse

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["test", "brains"], required=True)
    parser.add_argument("--brain", choices=["bear", "chop", "bull"])
    parser.add_argument("--tag")
    parser.add_argument("--out")
    parser.add_argument("--coins")
    parser.add_argument("--start")
    parser.add_argument("--end")
    parser.add_argument("--no_write", action="store_true")
    parser.add_argument("--min_events", type=int)
    parser.add_argument("-v", dest="verbose", action="count", default=0)
    args = parser.parse_args()

    if args.mode == "test":
        from systems import brain_score
        brain_score.run_single_brain(args)
    elif args.mode == "brains":
        from systems import brain_score
        brain_score.main(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()

