
import threading
try:
    import msvcrt
except ImportError:  # pragma: no cover - Windows-only
    msvcrt = None
import json
from tqdm import tqdm
from systems.utils.path import find_project_root
from systems.utils.logger import (
    addlog,
    init_logger,
    LOGGING_ENABLED,
)
from systems.scripts.get_candle_data import get_candle_data_df
from systems.scripts.get_window_data import get_window_data_df
from systems.scripts.evaluate_buy import evaluate_buy_df
from systems.scripts.evaluate_sell import evaluate_sell_df
import pandas as pd
from systems.utils.time import duration_from_candle_count
from systems.utils.settings_loader import load_settings

SETTINGS = load_settings()


def listen_for_keys(should_exit_flag: list) -> None:
    if not msvcrt:
        return
    while True:
        if msvcrt and msvcrt.kbhit():
            key = msvcrt.getch()
            if key == b'\x1b':  # ESC
                if not should_exit_flag:
                    should_exit_flag.append(True)
                    break

def run_simulation(tag: str, window: str, verbose: int = 0) -> None:
    init_logger(
        logging_enabled=LOGGING_ENABLED,
        verbose_level=verbose,
        telegram_enabled=False,
    )
    addlog(
        f"[SIM] Running simulation for {tag} on window {window}",
        verbose_int=1,
        verbose_state=verbose,
    )

    # Resolve exchange symbols (kraken/binance) for future use
    from systems.utils.resolve_symbol import resolve_symbol
    symbols = resolve_symbol(tag)

    from systems.scripts.ledger import RamLedger
    ledger = RamLedger()

    settings_path = find_project_root() / "settings" / "settings.json"
    with open(settings_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    sim_capital = float(config.get("simulation_capital", 0))
    start_capital = sim_capital

    def get_capital():
        return sim_capital

    def deduct_capital(note):
        nonlocal sim_capital
        sim_capital -= note.get("entry_usdt", 0.0)

    def credit_capital(note):
        nonlocal sim_capital
        sim_capital += note.get("exit_usdt", 0.0)

    project_root = find_project_root()
    data_path = project_root / "data/raw" / f"{tag}.csv"
    df = pd.read_csv(data_path)

    # Cache candle and window data
    total_rows = len(df)
    precomputed_candles = [get_candle_data_df(df, row_offset=i) for i in range(total_rows)]
    precomputed_windows = [get_window_data_df(df, window, candle_offset=i) for i in range(total_rows)]

    cooldowns = {
        "knife_catch": SETTINGS["general_settings"]["knife_catch_cooldown"],
        "whale_catch": SETTINGS["general_settings"]["whale_catch_cooldown"],
        "fish_catch": SETTINGS["general_settings"]["fish_catch_cooldown"]
    }

    last_triggered = {
        "knife_catch": None,
        "whale_catch": None,
        "fish_catch": None
    }

    should_exit = []

    # Start ESC listener
    listener_thread = threading.Thread(target=listen_for_keys, args=(should_exit,), daemon=True)
    listener_thread.start()

    with tqdm(total=total_rows, desc="📉 Sim Progress", dynamic_ncols=True) as pbar:
        for step in range(total_rows):
            if should_exit:
                addlog(
                    "\n🚪 ESC detected — exiting simulation early.",
                    verbose_int=1,
                    verbose_state=verbose,
                )

                # ✅ Get and print time range
                elapsed = duration_from_candle_count(step + 1, candle_interval_minutes=60)
                addlog(
                    f"⏱️  Simulated Range: {elapsed} ({step + 1} ticks of {total_rows})",
                    verbose_int=1,
                    verbose_state=verbose,
                )

                # ✅ Save and show ledger summary
                save_ledger_to_file(ledger, verbose=verbose)

                open_notes = ledger.get_active_notes()
                if open_notes:
                    last_price = precomputed_candles[step]["close"]
                    unrealized_value = sum(n["entry_amount"] * last_price for n in open_notes)
                    addlog(
                        f"[INFO] Mark-to-market added from open notes: ${unrealized_value:.2f}",
                        verbose_int=1,
                        verbose_state=verbose,
                    )
                    ending_capital = sim_capital + unrealized_value
                else:
                    ending_capital = sim_capital

                print_simulation_summary(
                    ledger,
                    starting_capital=start_capital,
                    ending_capital=ending_capital,
                    ticks_run=step + 1,
                    verbose=verbose,
                )
                break



            candle = precomputed_candles[step]
            window_data = precomputed_windows[step]

            if candle and window_data:
                evaluate_buy_df(
                    candle=candle,
                    window_data=window_data,
                    tick=step,
                    cooldowns=cooldowns,
                    last_triggered=last_triggered,
                    tag=tag,
                    sim=True,
                    verbose=verbose,
                    ledger=ledger,  # ✅ Inject ledger
                    get_capital=get_capital,
                    on_buy=deduct_capital,
                )

                to_sell = evaluate_sell_df(
                    candle=candle,
                    window_data=window_data,
                    tick=step,
                    notes=ledger.get_active_notes(),
                    tag=tag,
                    verbose=verbose,
                )
                
                exit_price = candle["close"]
                
                for note in to_sell:
                    
                    note["exit_price"] = exit_price
                    note["exit_ts"] = candle.get("ts", 0)
                    note["exit_tick"] = step
                    note["exit_usdt"] = exit_price * note["entry_amount"]
                    note["gain_pct"] = (note["exit_usdt"] - note["entry_usdt"]) / note["entry_usdt"]
                    note["status"] = "Closed"

                    ledger.close_note(note)
                    credit_capital(note)
                    addlog(
                        f"[SELL] Tick {step} | Strategy: {note['strategy']} | Gain: {note.get('gain_pct', 0):.2%}",
                        verbose_int=1,
                        verbose_state=verbose,
                    )
            else:
                addlog(
                    f"[STEP {step+1}] ❌ Incomplete data (candle or window)",
                    verbose_int=1,
                    verbose_state=verbose,
                )

            pbar.update(1)

    # End of simulation loop
    save_ledger_to_file(ledger, verbose=verbose)

    open_notes = ledger.get_active_notes()
    if open_notes:
        last_price = precomputed_candles[-1]["close"]
        unrealized_value = sum(n["entry_amount"] * last_price for n in open_notes)
        addlog(
            f"[INFO] Mark-to-market added from open notes: ${unrealized_value:.2f}",
            verbose_int=1,
            verbose_state=verbose,
        )
        ending_capital = sim_capital + unrealized_value
    else:
        ending_capital = sim_capital

    print_simulation_summary(
        ledger,
        starting_capital=start_capital,
        ending_capital=ending_capital,
        ticks_run=total_rows,
        verbose=verbose,
    )

def save_ledger_to_file(ledger, filename="ledgersimulation.json", verbose: int = 0) -> None:
    """Save the current state of a RamLedger to /data/filename.json"""
    import json
    from pathlib import Path
    from systems.utils.path import find_project_root

    output_path = Path(find_project_root()) / "data/tmp" / filename
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        json.dump({
            "open_notes": ledger.get_active_notes(),
            "closed_notes": ledger.get_closed_notes(),
            "summary": ledger.get_summary()
        }, f, indent=2)

    from tqdm import tqdm
    addlog(
        f"\n🧾 Ledger saved to: {output_path}",
        verbose_int=1,
        verbose_state=verbose,
    )
    
def print_simulation_summary(
    ledger,
    starting_capital=None,
    ending_capital=None,
    ticks_run=None,
    candle_minutes=60,
    verbose: int = 0,
) -> None:
    summary = ledger.get_summary()
    strategy_counts = ledger.get_trade_counts_by_strategy()

    output = ["📊 Simulation Summary"]

    if starting_capital is not None and ending_capital is not None:
        pnl = ending_capital - starting_capital
        pct = (pnl / starting_capital) * 100 if starting_capital else 0
        output += [
            f"Starting Capital: ${starting_capital:.2f}",
            f"Ending Capital:   ${ending_capital:.2f}",
            f"Net PnL:          ${pnl:.2f} ({pct:.2f}%)",
        ]
    else:
        output += [
            f"Open Notes:     {summary['num_open']}",
            f"Closed Notes:   {summary['num_closed']}",
            f"Investment:     ${summary['total_invested_usdt']:.2f}",
            f"Net PnL:        ${summary['total_pnl_usdt']:.2f}",
        ]

    output.append(f"Avg Gain %:     {summary['total_gain_pct']:.2%}")
    output.append("\n🎣 Strategy Breakdown")

    for strategy in ["knife_catch", "whale_catch", "fish_catch"]:
        data = strategy_counts.get(strategy, {"total": 0, "open": 0})
        output.append(
            f"{strategy.replace('_', ' ').title():<15}: {data['total']} trades ({data['open']} open)"
        )

    if ticks_run:
        gain_per_month = ledger.get_avg_gain_per_month(ticks_run, candle_minutes)
        roi_per_month = ledger.get_roi_per_month(ticks_run, candle_minutes)
        output += [
            f"Avg Gain %/mo:  {gain_per_month:.2%}",
            f"Avg ROI %/mo:   {roi_per_month:.2%}",
        ]

    addlog("\n" + "\n".join(output), verbose_int=0, verbose_state=verbose)
