
import threading
try:
    import msvcrt
except ImportError:  # pragma: no cover - Windows-only
    msvcrt = None
import json
from tqdm import tqdm
from systems.utils.path import find_project_root
from systems.scripts.get_candle_data import get_candle_data_df
from systems.scripts.get_window_data import get_window_data_df
from systems.scripts.evaluate_buy import evaluate_buy_df
from systems.scripts.evaluate_sell import evaluate_sell_df
import pandas as pd
from systems.utils.time import duration_from_candle_count


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
    if verbose >= 1:
        print(f"[SIM] Running simulation for {tag} on window {window}")

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
        "knife_catch": 1,
        "whale_catch": 0,
        "fish_catch": 2
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
                if verbose >= 1:
                    tqdm.write("\n🚪 ESC detected — exiting simulation early.")

                # ✅ Get and print time range
                elapsed = duration_from_candle_count(step + 1, candle_interval_minutes=60)
                if verbose >= 1:
                    tqdm.write(f"⏱️  Simulated Range: {elapsed} ({step + 1} ticks of {total_rows})")

                # ✅ Save and show ledger summary
                save_ledger_to_file(ledger, verbose=verbose)
                print_simulation_summary(
                    ledger,
                    starting_capital=start_capital,
                    ending_capital=sim_capital,
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
                    debug=False,
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
                    if verbose >= 1:
                        tqdm.write(
                            f"[SELL] Tick {step} | Strategy: {note['strategy']} | Gain: {note.get('gain_pct', 0):.2%}"
                        )
            else:
                if verbose >= 1:
                    tqdm.write(f"[STEP {step+1}] ❌ Incomplete data (candle or window)")

            pbar.update(1)

    # End of simulation loop
    save_ledger_to_file(ledger, verbose=verbose)
    print_simulation_summary(
        ledger,
        starting_capital=start_capital,
        ending_capital=sim_capital,
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
    if verbose >= 1:
        tqdm.write(f"\n🧾 Ledger saved to: {output_path}")
    
def print_simulation_summary(ledger, starting_capital=None, ending_capital=None, ticks_run=None, candle_minutes=60, verbose: int = 0) -> None:
    summary = ledger.get_summary()

    if verbose >= 1:
        tqdm.write("\n📊 Simulation Summary")
        if starting_capital is not None:
            tqdm.write(f"Starting Capital: ${starting_capital:.2f}")
        if ending_capital is not None:
            pnl = ending_capital - starting_capital if starting_capital is not None else 0
            pct = pnl / starting_capital * 100 if starting_capital else 0
            tqdm.write(f"Ending Capital:   ${ending_capital:.2f}")
            tqdm.write(f"Net PnL:          ${pnl:.2f} ({pct:.2f}%)")
        else:
            tqdm.write(f"Open Notes:     {summary['num_open']}")
            tqdm.write(f"Closed Notes:   {summary['num_closed']}")
            tqdm.write(f"Investment:     ${summary['total_invested_usdt']:.2f}")
            tqdm.write(f"Net PnL:        ${summary['total_pnl_usdt']:.2f}")
        tqdm.write(f"Avg Gain %:     {summary['total_gain_pct']:.2%}")
        tqdm.write(f"Est Balance:    ${summary['estimated_kraken_balance']:.2f}")

    strategy_counts = ledger.get_trade_counts_by_strategy()

    if verbose >= 1:
        tqdm.write("\n🎣 Strategy Breakdown")
        for strategy in ["knife_catch", "whale_catch", "fish_catch"]:
            data = strategy_counts.get(strategy, {"total": 0, "open": 0})
            total = data["total"]
            open_count = data["open"]
            tqdm.write(f"{strategy.replace('_', ' ').title():<15}: {total} trades ({open_count} open)")

    if ticks_run:
        gain_per_month = ledger.get_avg_gain_per_month(ticks_run, candle_minutes)
        roi_per_month = ledger.get_roi_per_month(ticks_run, candle_minutes)

        if verbose >= 1:
            tqdm.write(f"Avg Gain %/mo:  {gain_per_month:.2%}")
            tqdm.write(f"Avg ROI %/mo:   {roi_per_month:.2%}")

