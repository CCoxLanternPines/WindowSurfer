import time
import sys
import threading
from datetime import datetime, timedelta, timezone
from tqdm import tqdm

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


def run_live(tag: str, window: str) -> None:
    tqdm.write(f"[LIVE] Running live mode for {tag} on window {window}")
    should_exit = []

    if msvcrt:
        threading.Thread(target=esc_listener, args=(should_exit,), daemon=True).start()

    while not should_exit:
        now = datetime.utcnow().replace(tzinfo=timezone.utc)
        top_of_hour = (now + timedelta(hours=1)).replace(minute=0, second=0, microsecond=0)
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
                    tqdm.write("\n🚪 ESC detected — exiting live mode.")
                    return
                time.sleep(1)
                pbar.update(1)

        tqdm.write("\n🕐 Top of hour reached! Restarting countdown...\n")
