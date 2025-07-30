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
INVESTMENT_SIZE = SETTINGS.get("investment_size", 0.15)
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
    sim: bool = False,
    verbose: int = 0,
    ledger=None,
    get_capital=None,
    on_buy=None,
    meta=None  # ✅ THIS IS THE KEY FIX
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

    tunnel_pos = window_data.get("tunnel_position", 0)
    window_pos = window_data.get("window_position", 0)
    tunnel_high = window_data.get("window_ceiling", 0)
    tunnel_low = window_data.get("window_floor", 0)

    for key in cooldowns:
        cooldowns[key] = max(0, cooldowns[key] - 1)

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
        usd_amount = available_capital * INVESTMENT_SIZE
        if usd_amount < MINIMUM_NOTE_SIZE:
            min_size = MINIMUM_NOTE_SIZE
            addlog(
                f"[SKIP] Note below minimum size (${usd_amount:.2f} < ${min_size})",
                verbose_int=1,
                verbose_state=verbose,
            )
            return None

        # Deduct from local capital so sequential notes honor updates
        available_capital -= usd_amount

        entry_amount = usd_amount / close_price
        entry_usdt = usd_amount
        note = {
            "symbol": symbol,
            "strategy": strategy,
            "entry_price": close_price,
            "entry_ts": ts,
            "entry_tick": tick,
            "window": window_type,
            "entry_usdt": entry_usdt,
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
    
    # 🐟 Fish Catch
    if "fish_catch" in active and should_buy_fish(candle, window_data, tick, cooldowns):
        cooldowns["fish_catch"] = get_strategy_cooldown("fish_catch")
        last_triggered["fish_catch"] = tick if sim else ts
        note = create_note("fish_catch")
        if note:
            addlog(
                f"[BUY] Fish Catch triggered at tick {tick} → ${note['entry_usdt']:.2f}",
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
                    pair_code = meta["kraken_name"]
                    fiat = meta["fiat"]
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
                note["strategy"] = "fish_catch"
                ledger.open_note(note)
                if on_buy:
                    on_buy(note)
                triggered = True

    # 🐋 Whale Catch
    if "whale_catch" in active and should_buy_whale(candle, window_data, tick, cooldowns):
        cooldowns["whale_catch"] = get_strategy_cooldown("whale_catch")
        last_triggered["whale_catch"] = tick if sim else ts
        note = create_note("whale_catch")
        if note:
            addlog(
                f"[BUY] Whale Catch triggered at tick {tick} → ${note['entry_usdt']:.2f}",
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
                    pair_code = meta["kraken_name"]
                    fiat = meta["fiat"]
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
                note["strategy"] = "whale_catch"
                ledger.open_note(note)
                if on_buy:
                    on_buy(note)
                triggered = True

    # 🔪 Knife Catch
    if "knife_catch" in active and should_buy_knife(
        candle,
        window_data,
        tick,
        cooldowns,
        knife_notes,
        settings=SETTINGS,
        verbose=verbose,
    ):
        cooldowns["knife_catch"] = get_strategy_cooldown("knife_catch")
        last_triggered["knife_catch"] = tick if sim else ts
        note = create_note("knife_catch")
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
                    pair_code = meta["kraken_name"]
                    fiat = meta["fiat"]
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
                note["strategy"] = "knife_catch"
                ledger.open_note(note)
                if on_buy:
                    on_buy(note)
                triggered = True

    tunnel_height = tunnel_high - tunnel_low
    tunnel_pct = tunnel_pos * 100
    addlog(
        f"🧠 Tunnel {{w={tunnel_low:.4f}, h={tunnel_height:.4f}, p={tunnel_pos:.4f}, t={tunnel_pct:.1f}%}} "
        f"Window {{p={window_pos:.4f}}}",
        verbose_int=2,
        verbose_state=verbose,
    )

    fish_decision = should_buy_fish(candle, window_data, tick, cooldowns)
    whale_decision = should_buy_whale(candle, window_data, tick, cooldowns)
    knife_decision = should_buy_knife(
        candle,
        window_data,
        tick,
        cooldowns,
        knife_notes,
        settings=SETTINGS,
        verbose=verbose,
    )
    summary = f"[BUY RESULT] Tick {tick} | Knife: {knife_decision} | Whale: {whale_decision} | Fish: {fish_decision}"
    addlog(summary, verbose_int=2, verbose_state=verbose)


    return triggered
