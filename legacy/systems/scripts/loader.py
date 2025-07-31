import json
from pathlib import Path

from systems.utils.path import find_project_root


def load_settings():
    """Load settings.json from the project settings directory."""
    root = find_project_root()
    settings_path = root / "settings" / "settings.json"
    with open(settings_path, "r", encoding="utf-8") as f:
        return json.load(f)
