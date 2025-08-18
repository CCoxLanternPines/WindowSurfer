from __future__ import annotations

"""Helpers for formatting Telegram messages."""

from datetime import datetime, timezone
from typing import Optional


def describe_trend(slope: float) -> str:
    """Return human readable trend description from slope."""
    if slope > 0:
        prefix = "Strong" if slope > 0.3 else "Mild"
        return f"{prefix} Up"
    if slope < 0:
        prefix = "Strong" if slope < -0.3 else "Mild"
        return f"{prefix} Down"
    return "Flat"


def format_window_status(
    symbol: str,
    window: str,
    trend: str,
    slope: float,
    volatility: float,
    buy_trigger: float,
    current_pos: float,
    sell_trigger: float,
    open_notes: int,
    decision: str,
    action: Optional[str] = None,
    note: Optional[str] = None,
) -> str:
    """Return a multi-line Telegram status message for a window evaluation."""
    now = datetime.now(tz=timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    lines = [
        f"[WINDOW][{symbol}][{window}]",
        f"Time: {now}",
        f"Trend: {trend} | Slope: {slope:+.2f} | Volatility: {volatility:.2f}",
        f"Buy Trigger: {buy_trigger:.2f} | Current Pos: {current_pos:+.2f}",
        f"Sell Trigger: {sell_trigger:.2f} | Open Notes: {open_notes}",
        f"Decision: {decision}",
    ]
    if action:
        lines.append(f"Action: {action}")
    if note:
        lines.append(f"Note: {note}")
    return "\n".join(lines)
