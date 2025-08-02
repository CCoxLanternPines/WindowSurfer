"""Utility script to send a test message via the Telegram bot."""

from __future__ import annotations

from systems.utils.addlog import send_telegram_message, init_logger


if __name__ == "__main__":
    init_logger(telegram_enabled=True)
    send_telegram_message("✅ This is a test message from WindowSurfer bot.")
    print("Message sent.")

