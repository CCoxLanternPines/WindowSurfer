from __future__ import annotations

import json
from typing import Dict, Any

import requests

from .config import resolve_path

CACHE_FILE = resolve_path("data/snapshots/asset_pairs.json")


def load_asset_pairs() -> Dict[str, Any]:
    """Load Kraken AssetPairs from cache or live API.

    Returns
    -------
    dict
        Mapping of pair keys to info dictionaries as provided by Kraken.
    """
    if CACHE_FILE.exists():
        try:
            with CACHE_FILE.open("r", encoding="utf-8") as fh:
                return json.load(fh)
        except Exception:
            pass

    resp = requests.get("https://api.kraken.com/0/public/AssetPairs", timeout=10)
    data = resp.json().get("result", {})

    try:
        CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
        with CACHE_FILE.open("w", encoding="utf-8") as fh:
            json.dump(data, fh)
    except Exception:
        # Caching is best-effort; ignore failures.
        pass

    return data
