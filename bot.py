from __future__ import annotations

import argparse

from systems.walk_regime import run as run_regimes


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', required=True)
    parser.add_argument('--tag')
    parser.add_argument('--train')
    parser.add_argument('--test')
    parser.add_argument('--step')
    parser.add_argument('--clusters', type=int, default=4)
    parser.add_argument('--microtrials', type=int, default=0)
    parser.add_argument('--fees', type=float, default=0.0)
    parser.add_argument('--slip', type=float, default=0.0)
    parser.add_argument('--hysteresis', type=int, default=0)
    parser.add_argument('--blend', default='none')
    args = parser.parse_args(argv)

    if args.mode == 'regimes':
        if not all([args.tag, args.train, args.test, args.step]):
            parser.error('tag, train, test and step are required for regimes mode')
        run_regimes(
            tag=args.tag,
            train=args.train,
            test=args.test,
            step=args.step,
            clusters=args.clusters,
            microtrials=args.microtrials,
            fees=args.fees,
            slip=args.slip,
            hysteresis=args.hysteresis,
            blend=args.blend,
        )
    else:
        parser.error('--mode must be regimes')


if __name__ == '__main__':
    main()
