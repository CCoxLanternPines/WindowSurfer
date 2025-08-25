from __future__ import annotations
import os
from datetime import datetime
from pathlib import Path
from typing import Optional

# Verbosity semantics
WHAT  = 1  # "What I am doing"
WHY   = 2  # "Why I am doing it"
LOGIC = 3  # "Logic I am taking to make decisions"

_VERBOSITY = WHAT
_TO_FILE   = False
_FILE_PATH: Optional[Path] = None

def init_logger(*, verbosity: int = WHAT, to_file: bool = False, name_hint: str = "run") -> None:
    """
    Initialize global logger configuration.
    - verbosity: 1..3
    - to_file: write logs to data/temp/logs/<name_hint>_YYYYMMDD_HHMMSS.log
    """
    global _VERBOSITY, _TO_FILE, _FILE_PATH
    _VERBOSITY = max(WHAT, min(LOGIC, int(verbosity)))
    _TO_FILE   = bool(to_file)

    if _TO_FILE:
        log_dir = Path("data") / "temp" / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        _FILE_PATH = log_dir / f"{name_hint}_{ts}.log"
    else:
        _FILE_PATH = None

def log(level: int, message: str, to_file: Optional[bool] = None) -> None:
    """
    Log a message if level <= current verbosity.
    - level: 1 (WHAT), 2 (WHY), 3 (LOGIC)
    - to_file: override file output; default uses init_logger setting
    """
    if level > _VERBOSITY:
        return

    line = f"[L{level}] {message}"
    # Always echo to stdout for interactive runs
    print(line)

    # File output if enabled
    write_to_file = _TO_FILE if to_file is None else bool(to_file)
    if write_to_file and _FILE_PATH is not None:
        try:
            with _FILE_PATH.open("a", encoding="utf-8") as f:
                f.write(line + os.linesep)
        except Exception:
            # Never crash on logging failures
            pass

# Convenience wrappers
def what(msg: str):  log(WHAT,  msg)
def why(msg: str):   log(WHY,   msg)
def logic(msg: str): log(LOGIC, msg)
