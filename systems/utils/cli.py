import argparse


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
        "--ledger",
        help="[DEPRECATED] use --account instead",
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
