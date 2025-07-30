from __future__ import annotations

import json
from pathlib import Path


def find_project_root() -> Path:
    """Return project root by searching parent directories for marker files."""
    path = Path(__file__).resolve().parent
    for _ in range(5):
        if (path / "root").is_file():
            return path
        if path.parent == path:
            break
        path = path.parent
    raise FileNotFoundError("Project root not found using 'root' and 'systems' markers")


def load_settings():
    """Load settings.json from the project settings directory."""
    root = find_project_root()
    settings_path = root / "settings" / "settings.json"
    with open(settings_path, "r", encoding="utf-8") as f:
        return json.load(f)

