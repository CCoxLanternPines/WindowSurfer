import time
import sys
import threading
from datetime import datetime, timedelta, timezone
from tqdm import tqdm

from systems.scripts.get_candle_data import get_candle_data_json
from systems.scripts.get_window_data import get_window_data_json

try:
    import msvcrt  # Windows-only
except ImportError:
    msvcrt = None


def esc_listener(should_exit_flag):
    if not msvcrt:
        return
    while True:
        if msvcrt.kbhit():
            key = msvcrt.getch()
            if key == b'\x1b':  # ESC
                should_exit_flag.append(True)
                break


def run_live(tag: str, window: str, verbose: int = 0, debug: bool = False) -> None:
    if verbose:
        tqdm.write(f"[LIVE] Running live mode for {tag} on window {window}")
    should_exit = []

    if msvcrt:
        threading.Thread(target=esc_listener, args=(should_exit,), daemon=True).start()

    while not should_exit:
        if debug:
            if verbose:
                tqdm.write("[DEBUG] Skipping countdown, running top-of-hour logic now...")
            run_now = True
        else:
            now = datetime.utcnow().replace(tzinfo=timezone.utc)
            top_of_hour = (now + timedelta(hours=1)).replace(minute=0, second=0, microsecond=0)
            total_secs = 3600
            elapsed_secs = int(now.minute * 60 + now.second)
            remaining_secs = total_secs - elapsed_secs
            run_now = False
            total_secs = 3600
        elapsed_secs = int(now.minute * 60 + now.second)
        remaining_secs = total_secs - elapsed_secs

        with tqdm(
            total=total_secs,
            initial=elapsed_secs,
            desc="⏳ Time to next hour",
            bar_format="{l_bar}{bar}| {percentage:3.0f}% {remaining}s",
            leave=True,
            dynamic_ncols=True
        ) as pbar:
            for _ in range(remaining_secs):
                if should_exit:
                    if verbose:
                        tqdm.write("\n🚪 ESC detected — exiting live mode.")
                    return
                time.sleep(1)
                pbar.update(1)

        if verbose:
            tqdm.write("\n🕐 Top of hour reached! Restarting countdown...\n")

        candle = get_candle_data_json(tag, row_offset=0)
        window_data = get_window_data_json(tag, window, candle_offset=0)
        if candle and window_data:
            pass
