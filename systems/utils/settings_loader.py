import json
from systems.utils.path import find_project_root


def load_settings():
    """Load settings.json from the project settings directory."""
    root = find_project_root()
    settings_path = root / "settings" / "settings.json"
    with open(settings_path, "r", encoding="utf-8") as f:
        return json.load(f)


_SETTINGS = load_settings()


def get_strategy_cooldown(name: str) -> int:
    """Return cooldown value for a strategy from loaded settings."""
    return _SETTINGS.get("general_settings", {}).get(f"{name}_cooldown", 0)
