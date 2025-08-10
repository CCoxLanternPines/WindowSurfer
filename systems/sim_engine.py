from __future__ import annotations

from typing import Callable

from .utils.imports import ensure_project_root, import_module_safely

ensure_project_root()

_MODULE_CANDIDATES = [
    ("systems.simple_sim_engine", "systems/simple_sim_engine.py"),
    ("systems.engine", "systems/engine.py"),
    ("systems.scripts.simple_sim_engine", "systems/scripts/simple_sim_engine.py"),
    ("engine.simple_sim_engine", "engine/simple_sim_engine.py"),
    ("engine.sim_engine", "engine/sim_engine.py"),
]

_FUNC_CANDIDATES = [
    "run_sim",
    "run_simulation",
    "run_sim_blocks",
    "run_sim_engine",
]

_real_run_sim: Callable | None = None
for module_name, fallback_path in _MODULE_CANDIDATES:
    try:
        module = import_module_safely(module_name, fallback_path)
    except Exception:
        continue
    for func_name in _FUNC_CANDIDATES:
        func = getattr(module, func_name, None)
        if callable(func):
            _real_run_sim = func
            break
    if _real_run_sim:
        break

if _real_run_sim is None:
    searched_modules = ", ".join(name for name, _ in _MODULE_CANDIDATES)
    searched_funcs = ", ".join(_FUNC_CANDIDATES)
    raise ImportError(
        f"[sim_shim] Could not locate a sim runner. Looked for modules: {searched_modules} and functions: {searched_funcs}"
    )


def run_sim(*args, **kwargs):
    """Delegate to the real sim runner; returns dict with pnl, maxdd, trades, etc."""
    assert _real_run_sim is not None  # for type checkers
    return _real_run_sim(*args, **kwargs)


# Optional alias
run_simulation = run_sim
