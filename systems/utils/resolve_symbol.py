from systems.utils.settings_loader import load_settings

SETTINGS = load_settings()


def resolve_symbol(tag: str) -> dict:
    tag = tag.upper()
    entry = SETTINGS.get("symbol_settings", {}).get(tag)

    if not entry:
        raise ValueError(f"No settings found for tag: {tag}")

    return {
        "kraken": entry["kraken_name"],
        "binance": entry["binance_name"]
    }
