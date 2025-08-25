# WindowSurfer

WindowSurfer is a position-based cryptocurrency trading toolkit that operates on
rolling windows of historical candles. It can fetch market data, simulate
strategies, run them live against exchanges and inspect wallet balances.

## Entry Point

`bot.py` provides the command line interface and dispatches to the appropriate
subsystem based on `--mode`:

| Mode | Module | Description |
|------|--------|-------------|
| `fetch`  | `systems.fetch.run_fetch` | Sync hourly candles from Binance (simulation) and Kraken (live) for an account's ledger. |
| `sim`    | `systems.sim_engine.run_simulation` | Backtest the strategy on historical data. |
| `live`   | `systems.live_engine.run_live` | Execute the strategy on current market data; `--dry` runs once and exits. |
| `wallet` | `systems.scripts.wallet.show_wallet` | Display Kraken balances for the account's quote asset. |

Common options include `--account`, `-v/--verbose`, and `--log` to write
local JSON logs under `data/logs/`.

Examples:

```bash
python bot.py --mode fetch --account Kris
python bot.py --mode sim --account Kris -vv --time 7d
python bot.py --mode live --account Kris --dry
python bot.py --mode wallet --account Kris
python bot.py --mode test --account SAMPLE
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
- **execution_handler.py** – communicates with Kraken for live order placement.
- **fetch_candles.py** – low-level Binance/Kraken OHLCV downloaders used by
  `fetch` and cache utilities.
- **kraken_utils.py** – loads API keys and balances from Kraken.
- **wallet.py** – shows Kraken balances for a ledger's quote asset.

### systems/utils

Utility layer shared across modules:

- **addlog.py** – logging helper writing to `data/tmp/log.txt`.
- **config.py** – loads `settings/settings.json`, resolves project paths and
  warns about deprecated keys.
- **cli.py** – central argument parser defining supported modes and options.
- **resolve_symbol.py** – converts between human-friendly market strings and
  exchange-specific identifiers.
- **time.py** – helpers for parsing relative durations such as `7d` or `1m`.
- Additional helpers: `quote_norm.py`, `price_fetcher.py`, `trade_eval.py`.

### Configuration & Data

- **settings/settings.json** – retains ``general_settings`` such as capital limits.
- **settings/account_settings.json** – per-account flags and per-market sizing under
  ``"market settings"``.
- **settings/coin_settings.json** – coin-level strategy defaults with a ``default``
  block and optional per-symbol overrides.
- **settings/keys.json** – API key/secret pairs (git-ignored; provide your own).
- **Data directories**
  - `data/sim/` – historical candles used for simulations.
  - `data/live/` – live candles updated by `fetch` and the live engine.
  - `data/ledgers/` – JSON ledgers recording open/closed notes.
  - `data/tmp/` – transient files such as `log.txt` and simulation output.

Simulation reports may also be written to `logs/`.

### Additional Utilities

- `systems/manual.py` for manual test trades.
- `reference_logic/sim_engine.py` provides a compact reference
  implementation of the simulation algorithm.

## Installation

```bash
pip install -r requirements.txt
```

## Disclaimer

WindowSurfer is an experimental research toolkit. It is not audited and does
not guarantee profit; use at your own risk.
