import json
from pathlib import Path

from systems.utils import settings_loader


def test_simulation_writes_ledger(monkeypatch):
    import pandas as pd
    from systems import sim_engine

    # small dataset with three candles
    df = pd.DataFrame(
        {
            "timestamp": [1, 2, 3, 4],
            "close": [1.0, 2.0, 3.0, 4.0],
        }
    )

    def fake_loader(account, market, verbose=0, live=False):
        return df, 0

    def fake_buy(ctx, t, series, cfg=None, runtime_state=None):
        if t == 0:
            return {"size_usd": 10.0}
        return False

    def fake_sell(ctx, t, series, cfg=None, open_notes=None, runtime_state=None):
        if t == 1 and open_notes:
            return open_notes
        return []

    monkeypatch.setattr(sim_engine, "load_candles_df", fake_loader)
    monkeypatch.setattr(sim_engine, "evaluate_buy", fake_buy)
    monkeypatch.setattr(sim_engine, "evaluate_sell", fake_sell)
    monkeypatch.setattr(sim_engine, "resolve_symbols", lambda client, market: {"kraken_name": market, "kraken_pair": market, "binance_name": market})
    from systems.scripts import runtime_state
    monkeypatch.setattr(runtime_state, "resolve_symbols", lambda client, market: {"kraken_name": market, "kraken_pair": market, "binance_name": market})

    def fake_state(*args, **kwargs):
        return {
            "strategy": {
                "window_size": 1,
                "window_step": 1,
                "buy_trigger": 1,
                "max_pressure": 1,
                "investment_fraction": 1,
                "min_note_size": 0,
                "max_note_usdt": 1000,
            },
            "capital": 1000,
            "last_features": {},
            "pressures": {"buy": {"strategy": 0.0}, "sell": {"strategy": 0.0}},
        }

    monkeypatch.setattr(sim_engine, "build_runtime_state", fake_state)

    sim_engine.run_simulation(account="Kris", market="DOGEUSD", timeframe="100y", viz=False)

    ledger_path = Path("data/ledgers/ledger_simulation.json")
    assert ledger_path.exists()
    data = json.loads(ledger_path.read_text())
    sides = {e["side"] for e in data.get("entries", [])}
    assert {"BUY", "SELL", "PASS"}.issubset(sides)


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
