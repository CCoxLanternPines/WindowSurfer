from systems.utils.addlog import send_telegram_message


def notify_telegram(text: str) -> None:
    """Send a Telegram message if credentials are configured."""
    send_telegram_message(text)
