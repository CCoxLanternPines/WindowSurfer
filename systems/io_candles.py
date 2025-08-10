import csv
import os
from datetime import datetime, timezone
from typing import Optional, List, Tuple, Dict


def load_candles(tag: str, path_csv: Optional[str] = None, allow_gaps: bool = False) -> List[Tuple[datetime, Dict[str, float]]]:
    tag = tag.lower()
    if path_csv is None:
        base = tag.replace('usdt', 'usd')
        path_csv = os.path.join('data', 'raw', f'{base}.csv')
        if not os.path.exists(path_csv):
            alt = os.path.join('data', 'raw', f'{tag.split("usd")[0].upper()}.csv')
            if os.path.exists(alt):
                path_csv = alt
    if not os.path.exists(path_csv):
        raise FileNotFoundError(f'No candle file at {path_csv}')

    candles: List[Tuple[datetime, Dict[str, float]]] = []
    with open(path_csv, 'r', newline='') as f:
        reader = csv.DictReader(f)
        prev_ts = None
        expected_step = None
        for row in reader:
            ts = int(row['timestamp'])
            if prev_ts is not None:
                step = ts - prev_ts
                if expected_step is None:
                    expected_step = step
                else:
                    expected_step = min(expected_step, step)
                    if not allow_gaps and step > expected_step * 1000:
                        raise ValueError('Candle data has gaps')
            candle = {
                'open': float(row['open']),
                'high': float(row['high']),
                'low': float(row['low']),
                'close': float(row['close']),
                'volume': float(row['volume']),
            }
            candles.append((datetime.fromtimestamp(ts, tz=timezone.utc), candle))
            prev_ts = ts
    return candles
