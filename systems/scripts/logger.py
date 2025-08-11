import logging

logger = logging.getLogger("WindowSurfer")

if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
    logger.addHandler(handler)

# Allow bot.py verbosity setup to override this
logger.setLevel(logging.DEBUG)
logger.propagate = False
