import argparse
import importlib.util
from pathlib import Path
from typing import List, Dict

try:  # Optional dependency
    import numpy as np  # type: ignore
except Exception:  # pragma: no cover - environment fallback
    np = None
import pandas as pd


def load_data(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path, engine="python", on_bad_lines="skip")
    df.columns = [c.strip().lower() for c in df.columns]
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df[["timestamp", "open", "high", "low", "close", "volume"]]
    df = df.dropna(subset=["close"]).reset_index(drop=True)
    return df


def load_signal_modules(mod_dir: Path) -> List[Dict]:
    modules: List[Dict] = []
    if not mod_dir.exists():
        return modules
    for path in mod_dir.glob("*.py"):
        spec = importlib.util.spec_from_file_location(path.stem, path)
        module = importlib.util.module_from_spec(spec)
        try:
            spec.loader.exec_module(module)  # type: ignore
        except Exception as exc:  # pragma: no cover - best effort
            print(f"Failed to load {path.name}: {exc}")
            continue
        if not hasattr(module, "calculate"):
            continue
        name = getattr(module, "NAME", path.stem)
        lookback = getattr(module, "LOOKBACK", 0)
        modules.append({"name": name, "lookback": lookback, "module": module})
    return modules


def evaluate(df: pd.DataFrame, modules: List[Dict], lookahead: int, chunk: int, min_fires: int):
    results = []
    total_candles = max(len(df) - lookahead, 1)
    for info in modules:
        name = info["name"]
        mod = info["module"]
        lookback = info["lookback"]
        wins = 0
        fires = 0
        returns: List[float] = []
        for start in range(0, len(df), chunk):
            end = min(start + chunk, len(df))
            chunk_df = df.iloc[:end]
            for i in range(start, end - lookahead):
                if i < lookback:
                    continue
                try:
                    pred = mod.calculate(chunk_df, i)
                except Exception:  # pragma: no cover - best effort
                    pred = 0
                if pred not in (1, -1):
                    continue
                future_close = chunk_df["close"].iloc[i + lookahead]
                current_close = chunk_df["close"].iloc[i]
                ret = (future_close - current_close) / current_close
                if pred == -1:
                    ret = -ret
                returns.append(ret)
                fires += 1
                if ret > 0:
                    wins += 1
        if fires >= min_fires and fires > 0:
            results.append(
                {
                    "name": name,
                    "fires": fires,
                    "hit_rate": wins / fires,
                    "coverage": fires / total_candles,
                    "avg_return": float(np.mean(returns)) if returns and np else (sum(returns) / len(returns) if returns else 0.0),
                }
            )
    return results


def main():
    parser = argparse.ArgumentParser(description="Evaluate signal modules on historical data")
    parser.add_argument("--csv", required=True, help="Path to historical candle CSV")
    parser.add_argument("--lookahead", required=True, type=int, help="Candles ahead to check result")
    parser.add_argument("--chunk", type=int, default=500, help="Walk-forward chunk size")
    parser.add_argument("--min-fires", type=int, default=1, help="Minimum predictions required")
    args = parser.parse_args()

    base_dir = Path(__file__).resolve().parent.parent
    csv_path = Path(args.csv)
    if not csv_path.is_absolute():
        csv_path = base_dir / csv_path

    df = load_data(csv_path)
    modules = load_signal_modules(base_dir / "signal_modules")
    if not modules:
        print("No signal modules found")
        return

    results = evaluate(df, modules, args.lookahead, args.chunk, args.min_fires)
    if not results:
        print("No signals met criteria")
        return

    print(f"{'Signal':<12} {'Fires':>6} {'Hit%':>7} {'Coverage%':>11} {'AvgReturn%':>12}")
    for r in sorted(results, key=lambda x: x["hit_rate"], reverse=True):
        print(
            f"{r['name']:<12} {r['fires']:>6} {r['hit_rate']*100:>6.1f} {r['coverage']*100:>11.1f} {r['avg_return']*100:>12.2f}"
        )


if __name__ == "__main__":
    main()
