import os
from tqdm import tqdm

LOGFILE_PATH = "data/tmp/log.txt"
LOGGING_ENABLED = False
DEFAULT_VERBOSE_STATE = 1


def init_logger(logging_enabled: bool = False, verbose_level: int = 1) -> None:
    """Initialize logger settings."""
    global LOGGING_ENABLED, DEFAULT_VERBOSE_STATE
    LOGGING_ENABLED = logging_enabled
    DEFAULT_VERBOSE_STATE = verbose_level
    if LOGGING_ENABLED:
        os.makedirs("data/tmp", exist_ok=True)
        open(LOGFILE_PATH, "w").close()


def addlog(message: str, verbose_int: int = 1, verbose_state: int | None = None) -> None:
    """Write a log message if ``verbose_int`` is within ``verbose_state``."""
    if verbose_state is None:
        verbose_state = DEFAULT_VERBOSE_STATE
    if verbose_int <= verbose_state:
        tqdm.write(message)
    if LOGGING_ENABLED:
        with open(LOGFILE_PATH, "a", encoding="utf-8") as f:
            f.write(message + "\n")
