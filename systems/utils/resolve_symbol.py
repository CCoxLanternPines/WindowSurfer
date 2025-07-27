import json
from systems.utils.path import find_project_root


def resolve_symbol(tag: str) -> dict:
    tag = tag.upper()
    root = find_project_root()
    settings_path = root / "settings" / "settings.json"

    if not settings_path.exists():
        raise FileNotFoundError(f"settings.json not found at {settings_path}")

    with settings_path.open("r", encoding="utf-8") as f:
        config = json.load(f)

    symbol_settings = config.get("symbol_settings", {})
    entry = symbol_settings.get(tag)

    if not entry:
        raise ValueError(f"Tag '{tag}' not found in symbol_settings")

    return {
        "kraken": entry["kraken_name"],
        "binance": entry["binance_name"]
    }
