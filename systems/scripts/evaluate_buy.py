import json
from pathlib import Path
from systems.utils.path import find_project_root
from tqdm import tqdm

from systems.decision_logic.fish_catch import should_buy_fish
from systems.decision_logic.knife_catch import should_buy_knife
from systems.decision_logic.whale_catch import should_buy_whale
from systems.scripts.ledger import RamLedger

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
    sim: bool = False,
    verbose: bool = False,
    ledger=None  # <- Inject ledger if in RAM mode
) -> bool:
    """
    Evaluates buy conditions. Triggers notes via ledger if provided.
    Returns True if any buy was triggered.
    """
    if sim and not _log_initialized["sim"]:
        LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        open(LOG_PATH, "w").close()
        _log_initialized["sim"] = True
    
    tunnel_pos = window_data.get("tunnel_position", 0)
    window_pos = window_data.get("window_position", 0)
    tunnel_high = window_data.get("window_ceiling", 0)
    tunnel_low = window_data.get("window_floor", 0)

    for key in cooldowns:
        cooldowns[key] -= 1

    triggered = False
    close_price = candle["close"]
    ts = candle.get("ts", 0)  # or `None` if you want it explicit
    symbol = candle.get("symbol", "UNKNOWN")
    window_type = window_data.get("window", "1m")

    def create_note(strategy: str):
        entry_amount = 50.0
        entry_usdt = close_price * entry_amount
        return {
            "symbol": symbol,
            "strategy": strategy,
            "entry_price": close_price,
            "entry_ts": ts,
            "entry_tick": tick,
            "window": window_type,
            "entry_usdt": entry_usdt,
            "entry_amount": entry_amount,
            "status": "Open"
        }


    # 🐟 Fish Catch
    if should_buy_fish(candle, window_data, tick, cooldowns):
        cooldowns["fish_catch"] = 10
        last_triggered["fish_catch"] = tick
        tqdm.write(f"[BUY] Fish Catch triggered at tick {tick}")
        if ledger:
            ledger.add_note(create_note("fish_catch"))
        triggered = True

    # 🐋 Whale Catch
    if should_buy_whale(candle, window_data, tick, cooldowns):
        cooldowns["whale_catch"] = 5
        last_triggered["whale_catch"] = tick
        tqdm.write(f"[BUY] Whale Catch triggered at tick {tick}")
        if ledger:
            ledger.add_note(create_note("whale_catch"))
        triggered = True

    # 🔪 Knife Catch
    if should_buy_knife(candle, window_data, tick, cooldowns):
        cooldowns["knife_catch"] = 8
        last_triggered["knife_catch"] = tick
        tqdm.write(f"[BUY] Knife Catch triggered at tick {tick}")
        if ledger:
            ledger.add_note(create_note("knife_catch"))
        triggered = True

    if verbose:
        tunnel_height = tunnel_high - tunnel_low
        tunnel_pct = tunnel_pos * 100
        tqdm.write(
            f"🧠 Tunnel {{w={tunnel_low:.4f}, h={tunnel_height:.4f}, p={tunnel_pos:.4f}, t={tunnel_pct:.1f}%}} "
            f"Window {{p={window_pos:.4f}}}"
        )

    return triggered
