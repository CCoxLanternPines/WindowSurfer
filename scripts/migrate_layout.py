from pathlib import Path
from systems.paths import ensure_dirs, TEMP_DIR, RAW_DIR
import shutil
import time


def main():
    ensure_dirs()
    run_id = f"legacy_{time.strftime('%Y%m%d_%H%M%S')}"
    base = TEMP_DIR / run_id
    (base / "blocks").mkdir(parents=True, exist_ok=True)
    (base / "features").mkdir(parents=True, exist_ok=True)
    (base / "cluster").mkdir(parents=True, exist_ok=True)
    (base / "audit").mkdir(parents=True, exist_ok=True)

    moves = [
        ("logs", base / "audit"),
        ("features", base / "features"),
        ("audit", base / "audit"),
    ]
    for src, dst in moves:
        p = Path(src)
        if p.exists():
            for item in p.iterdir():
                shutil.move(str(item), str(dst))
            try:
                p.rmdir()
            except Exception:
                pass
    print(f"[MIGRATE] Legacy artifacts moved under {base}")

    hist = Path("data/historical")
    if hist.exists():
        RAW_DIR.mkdir(parents=True, exist_ok=True)
        for item in hist.iterdir():
            shutil.move(str(item), str(RAW_DIR))
        try:
            hist.rmdir()
        except Exception:
            pass


if __name__ == "__main__":
    main()

