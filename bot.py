import argparse
import logging
import os

from systems import chop_sim


def build_parser():
    parser = argparse.ArgumentParser(description="Trading bot entrypoint")
    parser.add_argument('--mode', required=True, help='mode to run')
    parser.add_argument('--tag', default='SOLUSDT', help='trading pair tag')
    parser.add_argument('--path-csv', default=None, help='override candle csv path')
    parser.add_argument('--allow-gaps', action='store_true', help='allow gaps in data')
    parser.add_argument('--window', type=int, default=300)
    parser.add_argument('--slope-gap', type=int, default=6)
    parser.add_argument('--chop-slope-eps', type=float, default=0.0008)
    parser.add_argument('--atr-min', type=float, default=0.004)
    parser.add_argument('--atr-max', type=float, default=0.020)
    parser.add_argument('--entry-dev-pct', type=float, default=0.010)
    parser.add_argument('--take-profit-pct', type=float, default=0.04)
    parser.add_argument('--stop-loss-pct', type=float, default=0.02)
    parser.add_argument('--max-hold-bars', type=int, default=72)
    parser.add_argument('--risk-frac', type=float, default=0.10)
    parser.add_argument('--max-concurrent', type=int, default=1)
    parser.add_argument('--buy-cooldown', type=int, default=12)
    parser.add_argument('--sell-cooldown', type=int, default=6)
    parser.add_argument('--start-capital', type=float, default=1000)
    parser.add_argument('-v', action='count', default=0, dest='verbose')
    parser.add_argument('--log-file', default=None)
    parser.add_argument('--report-extended', action='store_true')
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    log_level = logging.WARNING
    if args.verbose == 1:
        log_level = logging.INFO
    elif args.verbose >= 2:
        log_level = logging.DEBUG
    logging.basicConfig(level=log_level,
                        format='%(message)s',
                        handlers=[logging.StreamHandler()])
    if args.log_file:
        fh = logging.FileHandler(args.log_file)
        fh.setFormatter(logging.Formatter('%(message)s'))
        logging.getLogger().addHandler(fh)

    if args.mode == 'chop-sim':
        chop_sim.run(args)
    else:
        raise SystemExit(f'Unknown mode {args.mode}')


if __name__ == '__main__':
    main()
