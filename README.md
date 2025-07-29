# WindowSurfer

## Project Overview
WindowSurfer is a cryptocurrency trading simulator and live trading toolkit. It focuses on pattern-based strategies operating within "tunnel" windows derived from historical candle data. The system can run purely in simulation or attach to live data for parity testing. Strategies include:

- **Knife Catch** – triggers near the tunnel floor when downward momentum stalls.
- **Whale Catch** – acts on large dips in the lower tunnel region.
- **Fish Catch** – buys moderately low within the tunnel.

Each strategy places and manages trade "notes" that are tracked in a ledger to evaluate overall performance.

## System Architecture

``bot.py`` is the command line entrypoint. Depending on ``--mode`` it executes either the simulation or live engine:

- ``systems/sim_engine.py`` – step-by-step simulator using historical candles.
- ``systems/live_engine.py`` – skeleton for real-time operation.

Decision logic lives under ``systems/decision_logic`` with one module per strategy. Additional support scripts provide data access and evaluations:

- ``systems/scripts/get_candle_data.py`` – load individual candles.
- ``systems/scripts/get_window_data.py`` – compute tunnel/window metrics.
- ``systems/scripts/evaluate_buy.py`` – evaluate buy conditions and log notes.
- ``systems/scripts/evaluate_sell.py`` – check open notes for sell triggers.
- ``systems/scripts/ledger.py`` – RAM-based ledger for notes and summaries.
- Utility helpers in ``systems/utils`` for path and time parsing.

## How to Use
Install dependencies (pandas, tqdm, requests) and run ``bot.py`` with arguments:

```bash
# equivalent ways to control verbosity:
python bot.py --mode sim --window 1m -vv        # verbose level 2
python bot.py --mode live --tag SOLUSD --window 3mo --verbose 1 --telegram
```

CLI arguments:

- ``--mode`` – ``sim`` for simulation or ``live`` for live mode.
- ``--tag`` – trading pair symbol, e.g. ``DOGEUSD`` (default: ``DOGEUSD``).
- ``--window`` – time window for tunnel metrics such as ``1m`` or ``3mo``.
- ``-v``/``--verbose`` – verbosity level 0–3 (use ``-v``/``-vv``/``-vvv`` or ``--verbose N``).
- ``--log`` – write all output to ``data/tmp/log.txt``.
- ``--telegram`` – enable Telegram alerts (requires ``telegram.yaml``).

## Simulation Features
The simulator reads raw candle data from ``data/raw/<TAG>.csv`` and computes tunnel metrics for each step. For every candle tick it:

1. Evaluates buy strategies with cooldown tracking.
2. Records new notes in the ledger when a strategy triggers.
3. Checks existing notes against sell logic.
4. Updates the ledger with realized PnL and closed notes.

Results are stored in ``ledgersimulation.json`` under ``data/tmp``. Each note records its entry window position and originating strategy for later analysis.

## What the System Excels At

- Realistic, tick-level backtesting using actual market data.
- Clear separation of each trading strategy.
- Pattern-centric decisions rather than price prediction.
- Adjustable verbosity and the ability to stop simulations early.

## What It Is Not

- Not a predictive AI model.
- Does not attempt to maximize profit above structural rules.
- Not yet hardened for long unattended deployments.

