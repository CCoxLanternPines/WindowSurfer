from systems.utils.addlog import addlog


def jp_info(msg: str) -> None:
    """Log jackpot info-level messages (-v)."""
    addlog(msg, verbose_int=1)


def jp_debug(msg: str) -> None:
    """Log jackpot debug-level messages (-vv)."""
    addlog(msg, verbose_int=2)


def jp_trace(msg: str) -> None:
    """Log jackpot trace-level messages (-vvv)."""
    addlog(msg, verbose_int=3)
