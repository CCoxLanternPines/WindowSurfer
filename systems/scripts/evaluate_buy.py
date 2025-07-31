import json
from pathlib import Path
from systems.utils.path import find_project_root
from tqdm import tqdm
from systems.utils.logger import addlog

from systems.decision_logic.fish_catch import should_buy_fish
from systems.decision_logic.knife_catch import should_buy_knife
from systems.decision_logic.whale_catch import should_buy_whale
from systems.scripts.ledger import RamLedger
from systems.scripts.execution_handler import buy_order
from systems.scripts.kraken_utils import get_kraken_balance

from systems.utils.settings_loader import load_settings, get_strategy_cooldown
from systems.utils.resolve_symbol import resolve_symbol

SETTINGS = load_settings()
MINIMUM_NOTE_SIZE = SETTINGS.get("minimum_note_size", 0)

LOG_PATH = Path(find_project_root()) / "data" / "tmp" / "eval_buy_log.jsonl"
_log_initialized = {"sim": False}

STATE_PATH = Path(find_project_root()) / "data" / "tmp" / "eval_state.json"

def save_eval_state(cooldowns: dict, last_triggered: dict, tick: int) -> None:
    state = {
        "cooldowns": cooldowns,
        "last_triggered": last_triggered,
        "last_tick": tick
    }
    with open(STATE_PATH, "w") as f:
        json.dump(state, f)

def load_eval_state() -> dict:
    if STATE_PATH.exists():
        try:
            with open(STATE_PATH, "r") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}


def evaluate_buy_df(
    candle: dict,
    window_data: dict,
    tick: int,
    cooldowns: dict,
    last_triggered: dict,
    tag: str,
    strategy: str,
    sim: bool = False,
    verbose: int = 0,
    ledger=None,
    get_capital=None,
    on_buy=None,
    meta=None,  # ✅ THIS IS THE KEY FIX
    max_note_usdt: float = 999999,
) -> bool:


    """
    Evaluates buy conditions. Triggers notes via ledger if provided.
    Returns True if any buy was triggered.
    """
    if sim and not _log_initialized["sim"]:
        LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        open(LOG_PATH, "w").close()
        _log_initialized["sim"] = True

    if verbose >= 3:
        for strat in cooldowns:
            last = last_triggered.get(strat)
            addlog(
                f"[DEBUG] {strat}: cooldown={cooldowns[strat]} | last_triggered={last}",
                verbose_int=3,
                verbose_state=verbose,
            )

    assert strategy in {"knife_catch", "whale_catch", "fish_catch"}, f"Unknown strategy: {strategy}"

    tunnel_pos = window_data.get("tunnel_position", 0)
    window_pos = window_data.get("window_position", 0)
    tunnel_high = window_data.get("window_ceiling", 0)
    tunnel_low = window_data.get("window_floor", 0)

    triggered = False
    close_price = candle["close"]
    ts = candle.get("ts", candle.get("timestamp", 0))
    symbol = candle.get("symbol", "UNKNOWN")
    window_type = window_data.get("window", "1m")
    symbols = resolve_symbol(tag)
    kraken_symbol = symbols["kraken"]

    live = not sim

    # Fetch capital once at the start
    if live:
        kraken_bal = get_kraken_balance(verbose=verbose)
        fiat = meta.get("fiat", "ZUSD") if meta else "ZUSD"
        available_capital = float(kraken_bal.get(fiat, 0.0))
        addlog(
            f"[INFO] Available capital in {fiat}: ${available_capital:.2f}",
            verbose_int=2,
            verbose_state=verbose,
        )
    else:
        available_capital = float(get_capital() if get_capital else 0.0)

    if available_capital < MINIMUM_NOTE_SIZE:
        addlog(
            f"[SKIP] Not enough capital to trade (${available_capital:.2f} < ${MINIMUM_NOTE_SIZE:.2f})",
            verbose_int=1,
            verbose_state=verbose,
        )
        return False

    open_notes = ledger.get_active_notes() if ledger else []
    knife_notes = [n for n in open_notes if n.get("strategy") == "knife_catch"]

    def create_note(strategy: str):
        nonlocal available_capital
        base_strat = strategy.replace("_catch", "")
        investment_key = f"{base_strat}_investment_size"
        investment_size = meta.get(investment_key) if meta else None
        if investment_size is None:
            investment_size = SETTINGS["general_settings"].get(investment_key, 0.05)
        usd_amount = available_capital * investment_size
        entry_usdt = min(usd_amount, max_note_usdt)
        if entry_usdt < MINIMUM_NOTE_SIZE:
            min_size = MINIMUM_NOTE_SIZE
            addlog(
                f"[SKIP] Note below minimum size (${entry_usdt:.2f} < ${min_size})",
                verbose_int=1,
                verbose_state=verbose,
            )
            return None

        # Deduct from local capital so sequential notes honor updates
        available_capital -= entry_usdt

        entry_amount = entry_usdt / close_price
        note = {
            "symbol": tag,
            "strategy": strategy,
            "entry_price": close_price,
            "entry_ts": ts,
            "entry_tick": tick,
            "window": meta["window"] if meta else window_type,
            "entry_usdt": entry_usdt,          # ✅ REQUIRED — missing before
            "entry_amount": entry_amount,
            "status": "Open",
        }

        note["strategy"] = strategy  # ensure strategy explicitly recorded

        if strategy == "knife_catch":
            note["window_position_at_entry"] = window_pos


        return note

    active = SETTINGS.get(
        "active_strategies",
        ["fish_catch", "whale_catch", "knife_catch"],
    )

    if strategy not in active:
        return False

    if strategy == "fish_catch" and should_buy_fish(candle, window_data, tick, cooldowns):
        cooldowns[strategy] = get_strategy_cooldown(strategy)
        last_triggered[strategy] = tick if sim else ts
        note = create_note(strategy)
        if note:
            addlog(
                f"[BUY] Fish Catch triggered at tick {tick} → ${note['entry_usdt']:.2f}",
                verbose_int=1,
                verbose_state=verbose,
            )
            if ledger:
                if live:
                    pair_code = meta["kraken_name"] if meta else "UNKNOWN"
                    fiat = meta["fiat"] if meta else "ZUSD"
                    fills = buy_order(pair_code, fiat, note["entry_usdt"], verbose=verbose)
                    note["entry_price"] = fills["price"]
                    note["entry_amount"] = fills["volume"]
                    note["entry_usdt"] = fills["cost"]
                    note["fee"] = fills["fee"]
                    note["entry_ts"] = fills["timestamp"]
                    note["kraken_txid"] = fills["kraken_txid"]
                else:
                    note["entry_price"] = candle["close"]
                    note["entry_amount"] = note["entry_usdt"] / candle["close"]
                    note["entry_ts"] = candle.get("ts", 0)

                note["status"] = "Open"
                note["strategy"] = strategy
                ledger.open_note(note)
                if on_buy:
                    on_buy(note)
                triggered = True

    elif strategy == "whale_catch" and should_buy_whale(candle, window_data, tick, cooldowns):
        cooldowns[strategy] = get_strategy_cooldown(strategy)
        last_triggered[strategy] = tick if sim else ts
        note = create_note(strategy)
        if note:
            addlog(
                f"[BUY] Whale Catch triggered at tick {tick} → ${note['entry_usdt']:.2f}",
                verbose_int=1,
                verbose_state=verbose,
            )
            if ledger:
                if live:
                    pair_code = meta["kraken_name"] if meta else "UNKNOWN"
                    fiat = meta["fiat"] if meta else "ZUSD"
                    fills = buy_order(pair_code, fiat, note["entry_usdt"], verbose=verbose)
                    note["entry_price"] = fills["price"]
                    note["entry_amount"] = fills["volume"]
                    note["entry_usdt"] = fills["cost"]
                    note["fee"] = fills["fee"]
                    note["entry_ts"] = fills["timestamp"]
                    note["kraken_txid"] = fills["kraken_txid"]
                else:
                    note["entry_price"] = candle["close"]
                    note["entry_amount"] = note["entry_usdt"] / candle["close"]
                    note["entry_ts"] = candle.get("ts", 0)

                note["status"] = "Open"
                note["strategy"] = strategy
                ledger.open_note(note)
                if on_buy:
                    on_buy(note)
                triggered = True

    elif strategy == "knife_catch" and should_buy_knife(
        candle,
        window_data,
        tick,
        cooldowns,
        knife_notes,
        settings=SETTINGS,
        verbose=verbose,
    ):
        cooldowns[strategy] = get_strategy_cooldown(strategy)
        last_triggered[strategy] = tick if sim else ts
        note = create_note(strategy)
        if note:
            addlog(
                f"[BUY] Knife Catch triggered at tick {tick} → ${note['entry_usdt']:.2f}",
                verbose_int=1,
                verbose_state=verbose,
            )
            if ledger:
                if live:
                    addlog(
                        f"[EXEC] Live buy triggered for {tag}",
                        verbose_int=1,
                        verbose_state=verbose,
                    )
                    pair_code = meta["kraken_name"] if meta else "UNKNOWN"
                    fiat = meta["fiat"] if meta else "ZUSD"
                    fills = buy_order(pair_code, fiat, note["entry_usdt"], verbose=verbose)
                    note["entry_price"] = fills["price"]
                    note["entry_amount"] = fills["volume"]
                    note["entry_usdt"] = fills["cost"]
                    note["fee"] = fills["fee"]
                    note["entry_ts"] = fills["timestamp"]
                    note["kraken_txid"] = fills["kraken_txid"]
                else:
                    note["entry_price"] = candle["close"]
                    note["entry_amount"] = note["entry_usdt"] / candle["close"]
                    note["entry_ts"] = candle.get("ts", 0)

                note["status"] = "Open"
                note["strategy"] = strategy
                ledger.open_note(note)
                if on_buy:
                    on_buy(note)
                triggered = True

    tunnel_height = tunnel_high - tunnel_low
    tunnel_pct = tunnel_pos * 100
    addlog(
        f"🧠 Tunnel {{w={tunnel_low:.4f}, h={tunnel_height:.4f}, p={tunnel_pos:.4f}, t={tunnel_pct:.1f}%}} "
        f"Window {{p={window_pos:.4f}}}",
        verbose_int=3,
        verbose_state=verbose,
    )

    addlog(
        f"[BUY RESULT] Tick {tick} | Strategy: {strategy} | Triggered: {triggered}",
        verbose_int=2,
        verbose_state=verbose,
    )


    return triggered
