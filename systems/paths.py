from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
RESULTS_DIR = DATA_DIR / "results"
BRAINS_RESULTS_DIR = RESULTS_DIR / "brains"


def ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p


def brains_events_path(coin: str, brain: str) -> Path:
    return ensure_dir(BRAINS_RESULTS_DIR) / f"{coin}_{brain}_events.csv"


def brains_summary_path() -> Path:
    return ensure_dir(BRAINS_RESULTS_DIR) / "summary.csv"
