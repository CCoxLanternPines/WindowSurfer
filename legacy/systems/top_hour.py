"""Run top-of-hour logic for one or more symbols with state persistence."""

from __future__ import annotations

from datetime import datetime
import json
from pathlib import Path
import ccxt

from systems.utils.path import find_project_root
from systems.utils.logger import addlog
from systems.utils.top_hour_report import format_top_of_hour_report
from systems.utils.settings_loader import load_settings
from systems.live_engine import ensure_latest_candles, evaluate_live_tick
from systems.scripts.get_candle_data import get_candle_data_json
from systems.scripts.get_window_data import get_window_data_json
from systems.scripts.ledger import load_ledger, save_ledger
from systems.scripts.kraken_utils import get_kraken_balance

DEFAULT_WINDOW = "3d"

# ---------------------------------------------------------------------------
# State Persistence
# ---------------------------------------------------------------------------
_STATE_DIR = Path(find_project_root()) / "data" / "tmp"


def _state_path(tag: str) -> Path:
    """Return path for persisted state for ``tag``."""
    return _STATE_DIR / f"top_state_{tag}.json"


def load_top_state(tag: str) -> tuple[dict, dict]:
    """Load cooldowns and last_triggered for ``tag``."""
    path = _state_path(tag)
    if path.exists():
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
                return data.get("cooldowns", {}), data.get("last_triggered", {})
        except Exception:
            pass
    return {
        "knife_catch": 0,
        "whale_catch": 0,
        "fish_catch": 0,
    }, {
        "knife_catch": None,
        "whale_catch": None,
        "fish_catch": None,
    }


def save_top_state(tag: str, cooldowns: dict, last_triggered: dict) -> None:
    """Persist cooldowns and last_triggered values for ``tag``."""
    path = _state_path(tag)
    path.parent.mkdir(parents=True, exist_ok=True)
    state = {"cooldowns": cooldowns, "last_triggered": last_triggered}
    with open(path, "w", encoding="utf-8") as f:
        json.dump(state, f)


# ---------------------------------------------------------------------------
# Top-of-hour processing
# ---------------------------------------------------------------------------

def handle_top_of_hour(tag: str, window: str = DEFAULT_WINDOW, verbose: int = 0) -> None:
    """Execute one top-of-hour evaluation cycle for ``tag``."""
    settings = load_settings()
    meta = settings["symbol_settings"][tag]
    meta["window"] = window

    addlog(
        f"[TOP] Processing {tag} | Kraken: {meta['kraken_name']} | "
        f"Wallet: {meta['wallet_code']} | Fiat: {meta['fiat']} (verbose {verbose})",
        verbose_int=1,
        verbose_state=verbose,
    )

    ensure_latest_candles(tag, lookback="48h", verbose=verbose)
    candle = get_candle_data_json(tag, row_offset=0)
    window_data = get_window_data_json(tag, window, candle_offset=0)
    if not candle or not window_data:
        addlog("[ERROR] Missing candle or window data", verbose_int=1, verbose_state=verbose)
        return

    ledger = load_ledger(tag)
    cooldowns, last_triggered = load_top_state(tag)

    exchange = ccxt.kraken({"enableRateLimit": True})

    evaluate_live_tick(
        candle=candle,
        window_data=window_data,
        ledger=ledger,
        cooldowns=cooldowns,
        last_triggered=last_triggered,
        tag=tag,
        meta=meta,
        exchange=exchange,
        verbose=verbose,
    )

    save_ledger(tag, ledger)
    save_top_state(tag, cooldowns, last_triggered)

    kraken_balance = get_kraken_balance(verbose)
    fiat_asset = meta["fiat"]
    wallet_code = meta.get("wallet_code", meta["kraken_name"].replace("USD", ""))
    available_usd = float(kraken_balance.get(fiat_asset, 0.0))
    available_coin = float(kraken_balance.get(wallet_code, 0.0))
    coin_price = candle["close"]
    coin_balance_usd = available_coin * coin_price
    total_liquid_value = available_usd + coin_balance_usd

    triggered = {k.title(): v is not None for k, v in last_triggered.items()}
    notes = ledger.get_trade_counts_by_strategy()

    report = format_top_of_hour_report(
        tag,
        datetime.utcnow(),
        available_usd,
        coin_balance_usd,
        wallet_code,
        total_liquid_value,
        triggered,
        notes,
    )
    addlog(report, verbose_int=1, verbose_state=verbose)


def run_top_hour_all(tag: str | None = None, window: str = DEFAULT_WINDOW, verbose: int = 0) -> None:
    """Run ``handle_top_of_hour`` for all configured symbols or a single ``tag``."""
    settings = load_settings()
    symbols = settings.get("symbol_settings", {})
    tags = [tag.upper()] if tag else list(symbols.keys())

    for t in tags:
        if t not in symbols:
            addlog(f"[WARN] Unknown symbol tag: {t}", verbose_int=1, verbose_state=verbose)
            continue
        handle_top_of_hour(t, window=window, verbose=verbose)
