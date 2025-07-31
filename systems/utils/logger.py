import os
from typing import Optional

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover - fallback when tqdm missing
    class tqdm:  # type: ignore
        @staticmethod
        def write(msg: str) -> None:
            print(msg)


LOGFILE_PATH = "data/tmp/log.txt"
LOGGING_ENABLED = False
DEFAULT_VERBOSE_STATE = 1

TELEGRAM_ENABLED = False
TELEGRAM_TOKEN: Optional[str] = None
TELEGRAM_CHAT_ID: Optional[str] = None
_TELEGRAM_WARNED = False


def init_logger(
    logging_enabled: bool = False,
    verbose_level: int = 1,
    telegram_enabled: bool = False,
) -> None:
    """Initialize logger settings."""
    global LOGGING_ENABLED, DEFAULT_VERBOSE_STATE, TELEGRAM_ENABLED
    global TELEGRAM_TOKEN, TELEGRAM_CHAT_ID, _TELEGRAM_WARNED

    LOGGING_ENABLED = logging_enabled
    DEFAULT_VERBOSE_STATE = verbose_level
    TELEGRAM_ENABLED = telegram_enabled
    _TELEGRAM_WARNED = False

    if LOGGING_ENABLED:
        os.makedirs("data/tmp", exist_ok=True)
        open(LOGFILE_PATH, "w").close()

    if TELEGRAM_ENABLED:
        tqdm.write("[WARN] Telegram not initialized: yaml dependency missing")
        TELEGRAM_ENABLED = False
        _TELEGRAM_WARNED = True


def addlog(
    message: str, verbose_int: int = 1, verbose_state: int | None = None
) -> None:
    """Minimal logger respecting verbosity levels."""
    level = DEFAULT_VERBOSE_STATE if verbose_state is None else verbose_state
    if level >= verbose_int:
        print(message)
