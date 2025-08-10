from argparse import ArgumentParser


def add_verbose(p: ArgumentParser):
    p.add_argument("-v", "--verbose", action="count", default=0, help="Increase verbosity (-v, -vv, -vvv)")
    return p


def add_tag(p: ArgumentParser, required=True):
    p.add_argument("--tag", type=str, required=required, help="Symbol tag, e.g., SOLUSDT")
    return p


def add_run_id(p: ArgumentParser, required=False, default=None):
    p.add_argument("--run-id", type=str, required=required, default=default, help="Run identifier, e.g., regimes_fresh")
    return p


def add_action(p: ArgumentParser, choices):
    p.add_argument("--action", type=str, choices=choices, required=True, help="Action to perform")
    return p
