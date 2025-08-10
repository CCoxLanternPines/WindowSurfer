from __future__ import annotations
import importlib

# Try preferred engines in order; fall back to simple_sim_engine
_CANDIDATES = [
    ("systems.engine", ["run_sim_blocks", "run_sim", "run_simulation"]),
    ("systems.scripts.simple_sim_engine", ["run_sim_blocks"]),
    ("engine.sim_engine", ["run_sim_blocks", "run_sim"]),
    ("systems.simple_sim_engine", ["run_sim_blocks"]),  # our new fallback
]

def _load_runner():
    last_err = None
    for mod_name, fns in _CANDIDATES:
        try:
            mod = importlib.import_module(mod_name)
            for fn in fns:
                if hasattr(mod, fn):
                    return getattr(mod, fn)
        except Exception as e:
            last_err = e
            continue
    raise ImportError(f"[sim_shim] No sim runner found. Tried: {_CANDIDATES}\nLast error: {last_err}")

# public API the rest of the code can call
run_sim_blocks = _load_runner()
