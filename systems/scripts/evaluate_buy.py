import json
from pathlib import Path
from systems.utils.path import find_project_root
from tqdm import tqdm

from systems.decision_logic.fish_catch import should_buy_fish
from systems.decision_logic.knife_catch import should_buy_knife
from systems.decision_logic.whale_catch import should_buy_whale
from systems.scripts.ledger import RamLedger
from systems.scripts.execution_handler import buy_order

SETTINGS_PATH = Path(find_project_root()) / "settings" / "settings.json"
with open(SETTINGS_PATH, "r", encoding="utf-8") as f:
    SETTINGS = json.load(f)

INVESTMENT_SIZE = SETTINGS.get("investment_size", 0.15)
MINIMUM_NOTE_SIZE = SETTINGS.get("minimum_note_size", 0)
from systems.utils.resolve_symbol import resolve_symbol

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
    ledger=None,  # <- Inject ledger if in RAM mode
    debug: bool = False,
    get_capital=None,
    on_buy=None,
) -> bool:
    """
    Evaluates buy conditions. Triggers notes via ledger if provided.
    Returns True if any buy was triggered.
    """
    if sim and not _log_initialized["sim"]:
        LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        open(LOG_PATH, "w").close()
        _log_initialized["sim"] = True

    if verbose >= 2:
        tqdm.write(f"[EVAL] Evaluating Buy for {tag} 🐟🐋🔪")

    tunnel_pos = window_data.get("tunnel_position", 0)
    window_pos = window_data.get("window_position", 0)
    tunnel_high = window_data.get("window_ceiling", 0)
    tunnel_low = window_data.get("window_floor", 0)

    for key in cooldowns:
        cooldowns[key] -= 1

    triggered = False
    close_price = candle["close"]
    ts = candle.get("ts", 720)  # or `None` if you want it explicit
    symbol = candle.get("symbol", "UNKNOWN")
    window_type = window_data.get("window", "1m")
    symbols = resolve_symbol(tag)
    kraken_symbol = symbols["kraken"]

    live = not sim and not debug

    if live and ledger:
        if any(n.get("symbol") == symbol and n.get("status") == "Open" for n in ledger.get_active_notes()):
            if verbose >= 1:
                tqdm.write(f"[SKIP] Already have open note for {symbol} — skipping buy")
            return False

    def create_note(strategy: str):
        capital = get_capital() if get_capital else 0.0
        usd_amount = capital * INVESTMENT_SIZE
        if usd_amount < MINIMUM_NOTE_SIZE:
            if verbose >= 2:
                min_size = MINIMUM_NOTE_SIZE
                tqdm.write(
                    f"[SKIP] Note below minimum size (${usd_amount:.2f} < ${min_size})"
                )
            return None

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

        if strategy == "knife_catch":
            note["window_position_at_entry"] = window_pos

        return note
    
    # 🐟 Fish Catch
    if should_buy_fish(candle, window_data, tick, cooldowns):
        cooldowns["fish_catch"] = 4
        last_triggered["fish_catch"] = tick
        if verbose >= 1:
            tqdm.write(f"[BUY] Fish Catch triggered at tick {tick}")
        if ledger:
            note = create_note("fish_catch")
            if note:
                if live:
                    fills = buy_order(kraken_symbol, note["entry_usdt"])
                    note["entry_price"] = fills["price"]
                    note["entry_amount"] = fills["volume"]
                    note["entry_usdt"] = fills["cost"]
                    note["fee"] = fills["fee"]
                    note["entry_ts"] = fills["timestamp"]
                    note["kraken_txid"] = fills["kraken_txid"]
                ledger.add_note(note)
                if on_buy:
                    on_buy(note)
                triggered = True

    # 🐋 Whale Catch
    if should_buy_whale(candle, window_data, tick, cooldowns):
        cooldowns["whale_catch"] = 2
        last_triggered["whale_catch"] = tick
        if verbose >= 1:
            tqdm.write(f"[BUY] Whale Catch triggered at tick {tick}")
        if ledger:
            note = create_note("whale_catch")
            if note:
                if live:
                    fills = buy_order(kraken_symbol, note["entry_usdt"])
                    note["entry_price"] = fills["price"]
                    note["entry_amount"] = fills["volume"]
                    note["entry_usdt"] = fills["cost"]
                    note["fee"] = fills["fee"]
                    note["entry_ts"] = fills["timestamp"]
                    note["kraken_txid"] = fills["kraken_txid"]
                ledger.add_note(note)
                if on_buy:
                    on_buy(note)
                triggered = True

    # 🔪 Knife Catch
    if should_buy_knife(candle, window_data, tick, cooldowns):
        cooldowns["knife_catch"] = 1
        last_triggered["knife_catch"] = tick
        if verbose >= 1:
            tqdm.write(f"[BUY] Knife Catch triggered at tick {tick}")
        if ledger:
            note = create_note("knife_catch")
            if note:
                if live:
                    fills = buy_order(kraken_symbol, note["entry_usdt"])
                    note["entry_price"] = fills["price"]
                    note["entry_amount"] = fills["volume"]
                    note["entry_usdt"] = fills["cost"]
                    note["fee"] = fills["fee"]
                    note["entry_ts"] = fills["timestamp"]
                    note["kraken_txid"] = fills["kraken_txid"]
                ledger.add_note(note)
                if on_buy:
                    on_buy(note)
                triggered = True

    if verbose >= 3:
        tunnel_height = tunnel_high - tunnel_low
        tunnel_pct = tunnel_pos * 100
        tqdm.write(
            f"🧠 Tunnel {{w={tunnel_low:.4f}, h={tunnel_height:.4f}, p={tunnel_pos:.4f}, t={tunnel_pct:.1f}%}} "
            f"Window {{p={window_pos:.4f}}}"
        )

    if verbose >= 2:
        fish_decision = should_buy_fish(candle, window_data, tick, cooldowns)
        whale_decision = should_buy_whale(candle, window_data, tick, cooldowns)
        knife_decision = should_buy_knife(candle, window_data, tick, cooldowns)
        summary = f"[BUY RESULT] Tick {tick} | Knife: {knife_decision} | Whale: {whale_decision} | Fish: {fish_decision}"
        tqdm.write(summary)


    return triggered
