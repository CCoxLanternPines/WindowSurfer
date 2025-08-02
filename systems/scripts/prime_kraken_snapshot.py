from __future__ import annotations

"""Prime a Kraken API data snapshot for reuse within the current hour."""

import json
import time
from pathlib import Path

from systems.scripts.execution_handler import _kraken_request
from systems.utils.addlog import addlog
from systems.utils.path import find_project_root


def prime_kraken_snapshot(api_key: str, api_secret: str, ledger_name: str, verbose: int = 0) -> None:
    """Fetch balance and trades once and cache them for the hour.

    Parameters
    ----------
    api_key, api_secret:
        Kraken API credentials.
    ledger_name:
        Identifier for the ledger; used to name the snapshot file.
    verbose:
        Verbosity level for optional logging.
    """
    balance_resp = _kraken_request("Balance", {}, api_key, api_secret).get("result", {})
    trades_resp = _kraken_request(
        "TradesHistory",
        {"type": "all", "trades": True},
        api_key,
        api_secret,
    ).get("result", {})

    snapshot = {
        "timestamp": int(time.time()),
        "balance": balance_resp,
        "trades": trades_resp.get("trades", trades_resp),
    }

    root = find_project_root()
    snap_dir = root / "data" / "snapshots"
    snap_dir.mkdir(parents=True, exist_ok=True)
    path = snap_dir / f"{ledger_name}.json"
    with open(path, "w") as f:
        json.dump(snapshot, f)

    addlog(
        f"[SNAPSHOT] Cached Kraken balance and trades for {ledger_name}",
        verbose_int=3,
        verbose_state=verbose,
    )
