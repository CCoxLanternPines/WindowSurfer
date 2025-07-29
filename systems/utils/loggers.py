"""Basic shared logger instance for runtime output."""

import logging

logger = logging.getLogger("windowsurfer")

if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter("%(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
