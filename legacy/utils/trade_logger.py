import json
import os
from typing import Any, Dict, List

_log_path: str | None = None


def init_logger(ledger_name: str) -> None:
    """Initialize JSON log file for a given ledger."""
    global _log_path
    os.makedirs("data/logs", exist_ok=True)
    _log_path = os.path.join("data", "logs", f"{ledger_name}.json")
    if not os.path.exists(_log_path):
        with open(_log_path, "w", encoding="utf-8") as f:
            json.dump([], f)


def _load_events() -> List[Dict[str, Any]]:
    if not _log_path:
        raise RuntimeError("Logger not initialized")
    if os.path.exists(_log_path):
        with open(_log_path, "r", encoding="utf-8") as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError:
                data = []
    else:
        data = []
    return data


def record_event(event: Dict[str, Any]) -> None:
    """Append ``event`` to the current ledger's JSON log."""
    if not _log_path:
        raise RuntimeError("Logger not initialized")
    data = _load_events()
    data.append(event)
    with open(_log_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
