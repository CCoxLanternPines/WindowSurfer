from pathlib import Path
import time
import json


def load_settings():
    with open("settings.json", "r") as f:
        return json.load(f)


S = load_settings()["paths"]

DATA_ROOT = Path(S["data_root"])
RAW_DIR = Path(S["raw_dir"])
TEMP_DIR = Path(S["temp_dir"])
BRAINS_DIR = Path(S["brains_dir"])
RESULTS_DIR = Path(S["results_dir"])
LOGS_DIR = Path(S["logs_dir"])


def ensure_dirs():
    for p in [RAW_DIR, TEMP_DIR, BRAINS_DIR, RESULTS_DIR, LOGS_DIR]:
        Path(p).mkdir(parents=True, exist_ok=True)


def new_run_id(prefix: str = "run"):
    return f"{prefix}_{time.strftime('%Y%m%d_%H%M%S')}"


def temp_run_dir(run_id):
    return TEMP_DIR / run_id


def temp_blocks_dir(run_id):
    return temp_run_dir(run_id) / "blocks"


def temp_features_dir(run_id):
    return temp_run_dir(run_id) / "features"


def temp_cluster_dir(run_id):
    return temp_run_dir(run_id) / "cluster"


def temp_audit_dir(run_id):
    return temp_run_dir(run_id) / "audit"


def raw_parquet(tag):
    return RAW_DIR / f"{tag}_1h.parquet"


def results_csv(tag, run_id):
    return RESULTS_DIR / f"regime_walk_results_{tag}_{run_id}.csv"


def brain_json(tag):
    return BRAINS_DIR / f"brain_{tag}.json"


def log_file(tag, run_id):
    return LOGS_DIR / f"regimes_{tag}_{run_id}.log"

