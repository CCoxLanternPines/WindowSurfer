#!/usr/bin/env python3
from __future__ import annotations

"""Minimal plotting utilities for ledgers and candles."""

import json
from pathlib import Path

import matplotlib.pyplot as plt  # type: ignore
import pandas as pd


def plot(ledger_path: str, candles_path: str) -> None:
    """Plot candle close prices with optional trade markers."""
    df = pd.read_csv(candles_path)
    plt.plot(df["timestamp"], df["close"], label="Close", color="blue")
    try:
        with open(ledger_path, "r", encoding="utf-8") as f:
            ledger = json.load(f)
        for note in ledger.get("closed_notes", []):
            if "created_ts" in note and "entry_price" in note:
                plt.scatter(note["created_ts"], note["entry_price"], color="green", marker="^")
            if "exit_ts" in note and "exit_price" in note:
                plt.scatter(note["exit_ts"], note["exit_price"], color="red", marker="v")
    except FileNotFoundError:
        pass
    plt.legend()
    plt.tight_layout()
    plt.show()
