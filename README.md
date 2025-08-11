# WindowSurfer

## Project Overview
Ledger-based multi-window crypto trading bot with sim/live parity. Uses tunnel strategy with configurable buy/sell rules and WTF scaling.

## Features
- Multi-window (day/week/month/four_month)
- Per-ledger trading config
- WTF scaling & partial sell support
- Fetch full history (Binance) + live updates (Kraken)
- Wallet cache for exchange pair metadata
- Sim/live/dry-run modes

## Installation
```bash
git clone <repo>
cd <repo>
pip install -r requirements.txt
```

## Configuration
- `config/global_settings.yaml`
- `ledger/settings/{ledger_name}.json`

Defines coins, per-tunnel settings, and per-ledger trading rules.
Run `fetch.py --wallet_cache` before the first sim/live run.

## Usage
### Simulation
```bash
python bot.py --mode sim --ledger GoatLedger --start 1y --range 3m -vv
```
### Live Trading
```bash
python bot.py --mode live --ledger GoatLedger -vv
```
### Dry Run Live
```bash
python bot.py --mode live --ledger GoatLedger --dry-run -vv
```
### Data Fetch
```bash
python bot.py --mode fetch --ledger GoatLedger --full
python bot.py --mode fetch --ledger GoatLedger --update
python bot.py --mode fetch --wallet_cache
```

## Data Storage
- `data/raw` → historical data per coin
- `data/meta` → wallet cache JSON
- `data/temp` → sim ledger/temp state

## Notes
- UTC timestamps for all ops; convert to CTZ for queries
- Sim/live parity guaranteed
