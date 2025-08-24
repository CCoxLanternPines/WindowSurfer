import json
from pathlib import Path

from systems.utils import settings_loader


def test_load_coin_settings_overrides_and_defaults(tmp_path, monkeypatch):
    data = {
        "coin_settings": {
            "default": {"buy_trigger": 1, "window_size": 10},
            "DOGE/USD": {"buy_trigger": 4},
        }
    }
    cfg_file = tmp_path / "coin_settings.json"
    cfg_file.write_text(json.dumps(data))
    monkeypatch.setattr(settings_loader, "COIN_SETTINGS_PATH", cfg_file)

    cfg = settings_loader.load_coin_settings("DOGE/USD")
    assert cfg["buy_trigger"] == 4
    assert cfg["window_size"] == 10
