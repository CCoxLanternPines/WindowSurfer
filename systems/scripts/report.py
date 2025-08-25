from __future__ import annotations

"""Generate and email trading reports."""

import argparse
import json
import smtplib
import ssl
from datetime import datetime
from email.message import EmailMessage
from pathlib import Path

import yaml

from systems.scripts.view_log import export_png
from systems.scripts.ledger import load_ledger
from systems.utils.addlog import addlog
from systems.utils.config import load_settings


def run_report(account: str, period: str) -> None:
    """Create and email a report for ``account`` over ``period``."""

    cfg = load_settings()
    acct_cfg = cfg.get("accounts", {}).get(account)
    if not acct_cfg or not acct_cfg.get("is_live", False):
        addlog(f"[EMAIL][SKIP] {account} not live or unknown")
        return

    general_email = cfg.get("general_settings", {}).get("email", {})
    acct_reporting = acct_cfg.get("reporting", {})

    if period != "test":
        if not (general_email.get("enabled") and general_email.get(period, False)):
            addlog(f"[EMAIL][SKIP] {period} disabled")
            return
        if not acct_reporting.get(period, False):
            addlog(f"[EMAIL][SKIP] {account} {period} disabled")
            return
        now = datetime.utcnow()
        if period == "weekly" and now.weekday() != 0:
            addlog("[EMAIL][SKIP] not start of week")
            return
        if period == "monthly" and now.day != 1:
            addlog("[EMAIL][SKIP] not start of month")
            return
        if period == "yearly" and not (now.month == 1 and now.day == 1):
            addlog("[EMAIL][SKIP] not start of year")
            return

    log_path = Path(f"data/logs/{account}.json")
    if not log_path.exists():
        addlog("[EMAIL][SKIP] no log")
        return
    events = json.loads(log_path.read_text())
    if not events:
        addlog("[EMAIL][SKIP] empty log")
        return

    out_dir = log_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    png_path = out_dir / f"{account}_{period}.png"
    export_png(log_path, png_path)

    ledger = load_ledger(account)
    last_price = events[-1].get("features", {}).get("close", 0.0)
    summary = ledger.get_account_summary(last_price)
    buys = sum(1 for e in events if e.get("decision") == "BUY")
    sells = sum(1 for e in events if e.get("decision") == "SELL")
    flats = sum(1 for e in events if e.get("decision") == "FLAT")
    pnl = summary.get("realized_gain", 0.0)
    capital = ledger.get_metadata().get("capital", 0.0)
    roi = (pnl / capital) if capital else 0.0
    open_notes = summary.get("open_notes", 0)

    body = (
        f"PnL: ${pnl:.2f}\n"
        f"ROI: {roi*100:.2f}%\n"
        f"Buys: {buys} Sells: {sells} Flats: {flats}\n"
        f"Open notes: {open_notes}"
    )

    to_email = acct_reporting.get("email")
    if not to_email:
        addlog("[EMAIL][SKIP] no recipient")
        return

    key_path = Path("gmail_key.yaml")
    if not key_path.exists():
        addlog("[EMAIL][SKIP] no creds")
        return
    try:
        creds = yaml.safe_load(key_path.read_text())
    except Exception:
        addlog("[EMAIL][SKIP] no creds")
        return

    msg = EmailMessage()
    msg["Subject"] = f"{account} {period} report"
    msg["From"] = creds.get("email")
    msg["To"] = to_email
    msg.set_content(body)
    with png_path.open("rb") as fh:
        msg.add_attachment(fh.read(), maintype="image", subtype="png", filename="report.png")

    context = ssl.create_default_context()
    try:
        with smtplib.SMTP_SSL(creds["smtp_server"], creds["smtp_port"], context=context) as server:
            server.login(creds["email"], creds["app_password"])
            server.send_message(msg)
        addlog(f"[EMAIL][SENT] {account} {period} report to {to_email}")
    except Exception as exc:
        addlog(f"[EMAIL][FAIL] {exc}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--account", required=True)
    parser.add_argument(
        "--period",
        required=True,
        choices=["daily", "weekly", "monthly", "yearly", "test"],
    )
    args = parser.parse_args()
    run_report(args.account, args.period)


if __name__ == "__main__":
    main()
