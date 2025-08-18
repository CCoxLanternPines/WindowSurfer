# WindowSurfer

WindowSurfer is a position-based cryptocurrency trading toolkit that operates on
rolling windows of historical candles. It can fetch market data, simulate
strategies, run them live against exchanges and inspect wallet balances.

## Entry Point

`bot.py` provides the command line interface and dispatches to the appropriate
subsystem based on `--mode`:

| Mode | Module | Description |
|------|--------|-------------|
| `fetch`  | `systems.fetch.run_fetch` | Sync hourly candles from Binance (simulation) and Kraken (live) for a ledger. |
| `sim`    | `systems.sim_engine.run_simulation` | Backtest the strategy on historical data. |
| `live`   | `systems.live_engine.run_live` | Execute the strategy on current market data; `--dry` runs once and exits. |
| `wallet` | `systems.scripts.wallet.show_wallet` | Display Kraken balances for the ledger's quote asset. |

Common options include `--ledger`, `-v/--verbose`, `--log` to write
`data/tmp/log.txt` and `--telegram` to send alerts via `telegram.yaml`.

Examples:

```bash
python bot.py --mode fetch --ledger Kris_Ledger
python bot.py --mode sim --ledger Kris_Ledger -vv --time 7d
python bot.py --mode live --ledger Kris_Ledger --dry
python bot.py --mode wallet --ledger Kris_Ledger
```

## Architecture

### systems package

- **systems/fetch.py** – fetches hourly candles from Binance and Kraken and
  saves them under `data/sim/<TAG>_1h.csv` and `data/live/<TAG>_1h.csv`.
- **systems/sim_engine.py** – historical backtesting engine using window-based
  features and predictive pressures.
- **systems/live_engine.py** – mirrors the simulation logic against live
  candles, placing real orders when enabled.
- **systems/manual.py** – utility for manual Kraken buy/sell test orders outside
  the main bot.

### systems/scripts

Reusable components used by the engines:

- **evaluate_buy.py** – extracts window features and generates buy signals.
- **evaluate_sell.py** – decides which notes to exit based on pressures and
  current features.
- **runtime_state.py** – builds runtime state including capital, strategy
  defaults and limits for each ledger.
- **trade_apply.py** – applies buy and sell results to the ledger and adjusts
  capital.
- **ledger.py** – in-memory ledger with helpers to load/save trade notes to
  `data/ledgers` or simulation folders.
- **execution_handler.py** – communicates with Kraken for live order placement
  and snapshot loading.
- **candle_cache.py** – refreshes live candle caches and tracks historical
  high/low ranges.
- **fetch_candles.py** – low-level Binance/Kraken OHLCV downloaders used by
  `fetch` and cache utilities.
- **strategy_jackpot.py** – optional jackpot drip/cash-out logic that augments
  buy/sell behaviour.
- **kraken_utils.py** – loads API keys, snapshots and balances from Kraken.
- **wallet.py** – shows Kraken balances for a ledger's quote asset.
- `sim_tuner.py` – Optuna-based parameter tuner for simulations, run separately
  from the main bot.

### systems/utils

Utility layer shared across modules:

- **addlog.py** – logging helper with optional Telegram notifications writing to
  `data/tmp/log.txt`.
- **config.py** – loads `settings/settings.json`, resolves project paths and
  warns about deprecated keys.
- **cli.py** – central argument parser defining supported modes and options.
- **asset_pairs.py** – caches Kraken AssetPairs metadata for pair validation.
- **resolve_symbol.py** – converts between tags, exchange symbols and canonical
  data paths.
- **time.py** – helpers for parsing relative durations such as `7d` or `1m`.
- Additional helpers: `quote_norm.py`, `price_fetcher.py`, `telegram_utils.py`,
  `snapshot.py`, `trade_eval.py`.

### Configuration & Data

- **settings/settings.json** – defines ledgers and strategy defaults including
  window parameters and capital limits.
- **telegram.yaml** – optional Telegram bot token and chat id for alerts.
- **Data directories**
  - `data/sim/` – historical candles used for simulations.
  - `data/live/` – live candles updated by `fetch` and the live engine.
  - `data/ledgers/` – JSON ledgers recording open/closed notes.
  - `data/snapshots/` – cached exchange metadata (e.g. asset pairs).
  - `data/tmp/` – transient files such as `log.txt` and simulation output.

Simulation reports may also be written to `logs/`.

### Additional Utilities

- `systems/manual.py` for manual test trades.
- `systems/scripts/sim_tuner.py` for sequential parameter tuning.
- `reference_logic/sim_engine.py` provides a compact reference
  implementation of the simulation algorithm.

## Installation

```bash
pip install -r requirements.txt
```

## Disclaimer

WindowSurfer is an experimental research toolkit. It is not audited and does
not guarantee profit; use at your own risk.
