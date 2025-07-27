import time
import threading
import pandas as pd
from tqdm import tqdm

import sys
from systems.utils.path import find_project_root
sys.path.append(str(find_project_root()))

from systems.scripts.get_candle_data import get_candle_data
from systems.scripts.get_window_data import get_window_data
from systems.scripts.evaluate_buy import evaluate_buy

try:
    import msvcrt  # Windows only
except ImportError:
    msvcrt = None


def listen_for_keys(tick_delay, should_exit_flag):
    """Speed controls for +, -, =, Backspace, ESC"""
    if not msvcrt:
        return
    while True:
        if msvcrt.kbhit():
            key = msvcrt.getch()
            if key == b'+':
                tick_delay[0] = max(0.01, tick_delay[0] - 0.05)
            elif key == b'-':
                tick_delay[0] += 0.05
            elif key == b'=':
                tick_delay[0] = 0.15
            elif key == b'\x08':  # Backspace
                tick_delay[0] = 0.15
            elif key == b'\x1b':  # ESC
                if not should_exit_flag:
                    should_exit_flag.append(True)


def run_simulation(tag: str, window: str) -> None:
    tqdm.write(f"[SIM] Starting simulation for {tag} on window {window}")

    root = find_project_root()
    path = root / "data" / "raw" / f"{tag.upper()}.csv"
    df = pd.read_csv(path)
    total_rows = len(df)

    if total_rows == 0:
        tqdm.write("❌ No data in CSV.")
        return

    tick_delay = [0.15]
    should_exit = []

    if msvcrt:
        threading.Thread(target=listen_for_keys, args=(tick_delay, should_exit), daemon=True).start()

    with tqdm(
        total=total_rows,
        desc="📉 Sim Progress",
        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} rows",
        leave=True,
        dynamic_ncols=True
    ) as pbar:
        for step in range(total_rows):
            if should_exit:
                tqdm.write("\n🚪 ESC detected — exiting simulation early.")
                break

            candle = get_candle_data(tag, row_offset=step)
            window_data = get_window_data(tag, window, candle_offset=step)
            if candle and window_data:
                evaluate_buy(candle, window_data)
            else:
                 tqdm.write(f"[STEP {step+1}] ❌ Incomplete data (candle or window)")

            time.sleep(tick_delay[0])
            pbar.update(1)

    tqdm.write("\n✅ Simulation complete.")
