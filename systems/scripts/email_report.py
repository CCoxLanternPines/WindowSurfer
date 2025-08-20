from __future__ import annotations

"""Send scheduled trading reports via email."""

import json
import io
import smtplib
import ssl
from datetime import datetime, timedelta
from email.message import EmailMessage
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import yaml

from systems.utils.config import load_settings
from systems.scripts.view_log import view_log  # reuse plotting logic reference


def send_email(to_email: str, subject: str, body: str, image_bytes: bytes) -> None:
    """Send ``image_bytes`` as PNG attachment using Gmail SMTP credentials."""
    with open("gmail_key.yaml", "r", encoding="utf-8") as fh:
        creds = yaml.safe_load(fh)
    msg = EmailMessage()
    msg["Subject"] = subject
    msg["From"] = creds["email"]
    msg["To"] = to_email
    msg.set_content(body)
    msg.add_attachment(
        image_bytes,
        maintype="image",
        subtype="png",
        filename="report.png",
    )
    context = ssl.create_default_context()
    with smtplib.SMTP_SSL(
        creds["smtp_server"], creds["smtp_port"], context=context
    ) as server:
        server.login(creds["email"], creds["app_password"])
        server.send_message(msg)


def generate_report(ledger: str, timeframe: str) -> bytes | None:
    """Generate a trading chart for ``ledger`` and return PNG bytes."""
    log_path = Path(f"data/logs/{ledger}.json")
    if not log_path.exists():
        return None
    events = json.loads(log_path.read_text())
    if not events:
        return None
    now = datetime.utcnow()
    cutoff = {
        "day": now - timedelta(days=1),
        "week": now - timedelta(weeks=1),
        "month": now - timedelta(days=30),
        "year": now - timedelta(days=365),
    }[timeframe]
    df = pd.DataFrame(events)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df[df["timestamp"] >= cutoff]

    fig, ax = plt.subplots()
    buys = df[df["decision"] == "BUY"]
    sells = df[df["decision"] == "SELL"]
    flats = df[df["decision"] == "FLAT"]
    ax.scatter(
        buys["timestamp"],
        [t[0]["price"] for t in buys["trades"]],
        c="green",
        marker="^",
        label="Buy",
    )
    ax.scatter(
        sells["timestamp"],
        [t[0]["price"] for t in sells["trades"]],
        c="red",
        marker="v",
        label="Sell",
    )
    ax.scatter(
        flats["timestamp"],
        [t[0]["price"] for t in flats["trades"]],
        c="orange",
        marker="v",
        label="Flat",
    )
    ax.legend()
    ax.set_title(f"{ledger} Report ({timeframe})")
    ax.set_xlabel("Time")
    ax.set_ylabel("Price")
    plt.xticks(rotation=45)
    plt.tight_layout()

    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    plt.close(fig)
    buf.seek(0)
    return buf.getvalue()


def run_email_reports() -> None:
    """Generate and send email reports for configured accounts."""
    cfg = load_settings()
    for acct, acct_cfg in cfg.get("accounts", {}).items():
        reporting = acct_cfg.get("reporting", {})
        email = reporting.get("email")
        if not email:
            continue
        flags = [
            ("day", reporting.get("daily")),
            ("week", reporting.get("weekly")),
            ("month", reporting.get("monthly")),
            ("year", reporting.get("yearly")),
        ]
        for tf, enabled in flags:
            if enabled:
                img = generate_report(acct, tf)
                if img:
                    send_email(
                        email,
                        f"{acct} {tf} report",
                        "See attached chart.",
                        img,
                    )


if __name__ == "__main__":
    run_email_reports()
