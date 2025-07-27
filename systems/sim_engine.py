
import threading
import msvcrt
from tqdm import tqdm
from systems.utils.path import find_project_root
from systems.scripts.get_candle_data import get_candle_data_df
from systems.scripts.get_window_data import get_window_data_df
from systems.scripts.evaluate_buy import evaluate_buy_df
from systems.scripts.evaluate_sell import evaluate_sell_df
import pandas as pd

def listen_for_keys(should_exit_flag: list) -> None:
    if not msvcrt:
        return
    while True:
        if msvcrt.kbhit():
            key = msvcrt.getch()
            if key == b'\x1b':  # ESC
                if not should_exit_flag:
                    should_exit_flag.append(True)
                    
                    break

def run_simulation(tag: str, window: str, verbose: bool = False) -> None:
    print(f"[SIM] Running simulation for {tag} on window {window}")

    from systems.scripts.ledger import RamLedger
    ledger = RamLedger()

    project_root = find_project_root()
    data_path = project_root / "data/raw" / f"{tag}.csv"
    df = pd.read_csv(data_path)

    # Cache candle and window data
    total_rows = len(df)
    precomputed_candles = [get_candle_data_df(df, row_offset=i) for i in range(total_rows)]
    precomputed_windows = [get_window_data_df(df, window, candle_offset=i) for i in range(total_rows)]

    cooldowns = {
        "knife_catch": 0,
        "whale_catch": 0,
        "fish_catch": 0
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
                tqdm.write("\n🚪 ESC detected — exiting simulation early.")
                save_ledger_to_file(ledger)
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
                    sim=True,
                    verbose=verbose,
                    ledger=ledger  # ✅ Inject ledger
                )

                to_sell = evaluate_sell_df(
                    candle=candle,
                    window_data=window_data,
                    tick=step,
                    notes=ledger.get_active_notes(),
                    verbose=verbose,
                )
                for note in to_sell:
                    ledger.close_note(note)
                    tqdm.write(
                        f"[SELL] Tick {step} | Strategy: {note['strategy']} | Gain: {note.get('gain_pct', 0):.2%}"
                    )
            else:
                tqdm.write(f"[STEP {step+1}] ❌ Incomplete data (candle or window)")

            pbar.update(1)

def save_ledger_to_file(ledger, filename="ledgersimulation.json") -> None:
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
    tqdm.write(f"\n🧾 Ledger saved to: {output_path}")

