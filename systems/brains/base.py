from __future__ import annotations

from typing import Any, Dict


def default_init(settings: Dict[str, Any], ledger_cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Return a minimal state containing settings and ledger configuration."""
    return {"settings": settings, "ledger_cfg": ledger_cfg}
