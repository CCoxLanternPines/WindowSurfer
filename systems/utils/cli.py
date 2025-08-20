import argparse

from .addlog import addlog


def build_parser() -> argparse.ArgumentParser:
    """Return a configured argument parser for WindowSurfer CLI tools."""
    parser = argparse.ArgumentParser(description="WindowSurfer command line interface")
    parser.add_argument(
        "--mode",
        choices=["fetch", "sim", "live", "wallet"],
        help="Execution mode: fetch, sim, live, or wallet",
    )
    parser.add_argument(
        "--account",
        help="Account name defined in accounts.json",
    )
    parser.add_argument(
        "--market",
        help="Specific market symbol to operate on (default: all for account)",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Run all configured accounts and their markets",
    )
    parser.add_argument(
        "--ledger",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--dry",
        action="store_true",
        help="Run live mode once immediately and exit",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="count",
        default=0,
        help="Increase verbosity level (use -v or -vv)",
    )
    parser.add_argument(
        "--log",
        action="store_true",
        help="Enable log file output",
    )
    parser.add_argument(
        "--telegram",
        action="store_true",
        help="Enable Telegram alerts",
    )
    return parser


def handle_legacy_args(args: argparse.Namespace) -> argparse.Namespace:
    """Map deprecated ``--ledger`` flag to ``--account``."""
    if getattr(args, "ledger", None):
        addlog(
            "[DEPRECATED] --ledger is deprecated; use --account",
            verbose_int=1,
            verbose_state=True,
        )
        if not getattr(args, "account", None):
            args.account = args.ledger
    return args
