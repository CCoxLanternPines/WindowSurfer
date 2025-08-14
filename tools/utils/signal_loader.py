from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Callable, Dict, List

import pandas as pd


def load_signal_modules(mod_dir: Path) -> List[Dict[str, object]]:
    """Load all signal modules from *mod_dir*.

    Each module must define NAME, LOOKBACK and calculate(df, i).
    The returned list contains dictionaries with keys:
    ``name``, ``lookback`` and ``calculate`` (a leak-safe wrapper).
    """

    modules: List[Dict[str, object]] = []
    if not mod_dir.exists():
        return modules
    for path in sorted(mod_dir.glob("*.py")):
        spec = importlib.util.spec_from_file_location(path.stem, path)
        if spec is None or spec.loader is None:
            continue
        module = importlib.util.module_from_spec(spec)
        try:
            spec.loader.exec_module(module)  # type: ignore
        except Exception as exc:  # pragma: no cover - best effort
            print(f"Failed to load {path.name}: {exc}")
            continue
        if not hasattr(module, "calculate"):
            continue
        name = getattr(module, "NAME", path.stem)
        lookback = int(getattr(module, "LOOKBACK", 0))

        def _wrap(df: pd.DataFrame, i: int, _m=module) -> float:
            """Leak-safe wrapper calling module.calculate with df[:i+1]."""
            sub = df.iloc[: i + 1]
            try:
                return _m.calculate(sub, i)
            except Exception:  # pragma: no cover - best effort
                return 0.0

        modules.append({"name": name, "lookback": lookback, "calculate": _wrap})
    return modules
