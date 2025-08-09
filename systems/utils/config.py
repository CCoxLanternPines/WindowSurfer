from __future__ import annotations

"""Configuration helper utilities."""

from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[2]


def resolve_path(rel_path: str) -> Path:
    """Return an absolute path for ``rel_path`` from the project root."""
    return _PROJECT_ROOT / rel_path
