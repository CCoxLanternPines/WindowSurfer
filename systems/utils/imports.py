from __future__ import annotations
import sys
from pathlib import Path
import importlib
import importlib.util


def ensure_project_root():
    """
    Add the project root (the folder containing bot.py) to sys.path if missing.
    """
    # This file: .../systems/utils/imports.py
    utils_dir = Path(__file__).resolve().parent
    project_root = utils_dir.parent.parent  # go up to project root
    root_str = str(project_root)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)
    return project_root


def import_module_safely(module_name: str, fallback_rel_path: str | None = None):
    """
    Try normal import first. If it fails and a fallback path is provided,
    load the module directly from that path.
    """
    try:
        return importlib.import_module(module_name)
    except Exception:
        if not fallback_rel_path:
            raise
        project_root = ensure_project_root()
        full_path = (project_root / fallback_rel_path).resolve()
        if not full_path.exists():
            raise ImportError(f"Fallback path not found: {full_path}")
        spec = importlib.util.spec_from_file_location(module_name, full_path)
        mod = importlib.util.module_from_spec(spec)
        assert spec and spec.loader
        spec.loader.exec_module(mod)  # type: ignore[attr-defined]
        sys.modules[module_name] = mod
        return mod
