"""Helpers to load Kraken API credentials from configuration."""

from pathlib import Path
import yaml


def load_kraken_keys(path: str = "kraken.yaml") -> tuple[str, str]:
    """Load Kraken API key and secret from local YAML file."""
    file = Path(path)
    if not file.exists():
        raise FileNotFoundError("Missing kraken.yaml in project root")

    with file.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    kraken = data.get("kraken")
    if not kraken or "api_key" not in kraken or "api_secret" not in kraken:
        raise ValueError(
            "Malformed kraken.yaml: missing 'kraken.api_key' or 'api_secret'"
        )

    return kraken["api_key"], kraken["api_secret"]
