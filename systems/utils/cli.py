import argparse


def build_parser() -> argparse.ArgumentParser:
    """Return a configured argument parser for WindowSurfer CLI tools."""
    parser = argparse.ArgumentParser(description="WindowSurfer command line interface")
    parser.add_argument(
        "--mode",
        choices=["sim", "simtune", "live", "wallet"],
        help="Execution mode: sim, simtune, live, or wallet",
    )
    parser.add_argument(
        "--ledger",
        help="Ledger name defined in settings.json",
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
    parser.add_argument(
        "--jackpot",
        action="store_true",
        help="Enable jackpot DCA in simulation",
    )
    return parser
