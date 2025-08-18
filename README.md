# WindowSurfer

WindowSurfer is a position-based cryptocurrency trading toolkit. It focuses on pattern-driven strategies that act on the relative position of price inside rolling "windows" of historical candles. The system can simulate strategies on past data, tune parameters, run against fresh market data, inspect wallet balances and keep raw candle histories in sync.

## Goals
* Provide reproducible backtesting for window/"tunnel" strategies.
* Mirror simulation logic in live trading for apples-to-apples comparison.
* Maintain a ledger of trades ("notes") for later analysis and risk tracking.

## Modes
`bot.py` is the command-line entry point. Select behaviour with `--mode`:

| Mode | Purpose |
|------|---------|
| `sim` | Run a historical backtest using `systems/sim_engine.py`.
| `simtune` | Optimise strategy parameters with Optuna via `systems/scripts/sim_tuner.py`.
| `live` | Execute the same logic on fresh data at the top of every hour using `systems/live_engine.py`.
| `wallet` | Query Kraken for balances of the configured quote asset.
| `fetch` | Fill gaps in `data/raw/<TAG>.csv` by pulling candles from Kraken and Binance.

Examples:

```bash
python bot.py --mode fetch --ledger default --time 72h
python bot.py --mode sim --ledger default -vv
python bot.py --mode sim --ledger default --slow
python bot.py --mode live --ledger default --telegram
python bot.py --mode wallet --ledger default
```

## Architecture
* **bot.py** – parses CLI arguments, validates asset pairs and hands off to the proper engine.
* **systems/sim_engine.py** – iterates over historical candles, calling `evaluate_buy` and `evaluate_sell` to open or close notes and record metrics.
* **systems/live_engine.py** – mirrors the simulation logic against the most recent candle data; either runs once with `--dry` or loops each hour.
* **systems/fetch.py** – determines missing hours and retrieves them from Kraken or Binance.
* **systems/scripts/** – house the buy/sell evaluators, ledger implementation and data helpers.
* **systems/utils/** – configuration loader, logging/Telegram helpers, asset pair caching and CLI builder.

Trades are stored as *notes* in a JSON ledger. Each note records entry price, window metrics and targeted exit. The ledger can be persisted for live trading or inspected after simulations.

## Configuration
* `settings/settings.json` – defines one or more ledgers. Each ledger sets a trading pair (`tag`) and a collection of `window_settings` that describe strategy windows (size, trigger position, investment fraction, etc.).
* `telegram.yaml` – optional credentials for Telegram notifications.
* Candle files live under `data/raw/<TAG>.csv`; live mode updates ledgers under `data/ledgers/`.

## Logging & Alerts
Use `--log` to write output to `data/tmp/log.txt`. Passing `--telegram` enables notifications when `telegram.yaml` is present. Verbosity is controlled with `-v`/`-vv`.

## Installation
```bash
pip install -r requirements.txt
```

## Development Notes
The project currently ships without automated tests. Simulation results are written to `data/tmp/ledgersimulation.json`. Live mode waits until the next UTC hour between iterations while maintaining ledger state.

## Disclaimer
WindowSurfer is an experimental toolkit for research purposes. It does not guarantee profit and has not been audited for security or long-term unattended operation.
