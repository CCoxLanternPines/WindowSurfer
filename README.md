# WindowSurfer

WindowSurfer is a position-based cryptocurrency trading toolkit that operates on
rolling windows of historical candles. It can backtest strategies, run them live
against exchanges, and inspect logs.

## Entry Point

`bot.py` provides the command line interface and dispatches to the appropriate
subsystem based on `--mode`:

| Mode | Description |
|------|-------------|
| `sim`  | Backtest the strategy on historical data. |
| `live` | Execute the strategy on current market data; `--dry` runs once and exits. |
| `test` | Perform a forced micro buy and sell to validate API keys and trade flow. |
| `view` | Plot decision logs for an account. |

Common options include `--account`, `--market`, `-v/--verbose`, `--log` to write
`data/tmp/log.txt` and `--telegram` to send alerts via `telegram.yaml`.

Examples:

```bash
python bot.py --mode sim --account Kris --market DOGE/USD -vv --time 7d
python bot.py --mode live --account Kris --market DOGE/USD --dry
python bot.py --mode test --account Kris --market DOGE/USD
python bot.py --mode view --account Kris
```

## Configuration

Settings are split into:

- `settings/account_settings.json` – account-level configuration.
- `settings/coin_settings.json` – market definitions.
- `settings/settings.json` – general strategy defaults.

## Installation

```bash
pip install -r requirements.txt
```

## Disclaimer

WindowSurfer is an experimental research toolkit. It is not audited and does
not guarantee profit; use at your own risk.
