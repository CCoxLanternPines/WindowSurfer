from __future__ import annotations
from pathlib import Path
import json
from typing import Any, Dict

_CONFIG_PATH = Path(__file__).resolve().parents[2] / "settings" / "config.json"


def load_settings() -> Dict[str, Any]:
    """Load configuration from settings/config.json."""
    with _CONFIG_PATH.open("r", encoding="utf-8") as fh:
        data = json.load(fh)
    return data
