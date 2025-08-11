#!/usr/bin/env python3
from __future__ import annotations

"""Utility to migrate legacy <COIN><FIAT>.csv candle files to <COIN>.csv."""

import argparse
import shutil
from pathlib import Path

from systems.utils.config import load_settings, resolve_path


def migrate(apply: bool = False) -> None:
    settings = load_settings()
    root = resolve_path("")
    raw_dir = root / "data" / "raw"
    for name, cfg in settings.get("ledger_settings", {}).items():
        coin = cfg.get("coin")
        fiat = cfg.get("fiat")
        if not coin or not fiat:
            continue
        new_path = raw_dir / f"{coin.upper()}.csv"
        legacy_path = raw_dir / f"{(coin + fiat).upper()}.csv"
        if not new_path.exists() and legacy_path.exists():
            if apply:
                shutil.copyfile(legacy_path, new_path)
                print(f"Migrated {legacy_path.name} -> {new_path.name}")
            else:
                print(f"Would migrate {legacy_path.name} -> {new_path.name}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Migrate legacy candle files")
    parser.add_argument("--apply", action="store_true", help="Perform rename/copy")
    args = parser.parse_args()
    migrate(apply=args.apply)


if __name__ == "__main__":
    main()
