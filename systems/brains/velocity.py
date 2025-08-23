from __future__ import annotations

import json
from typing import Dict, List, Any

import numpy as np
import pandas as pd

from .base import Brain


class VelocityBrain(Brain):
    """Second brain that buckets direction and tracks angle velocity."""

    name = "velocity"

    def __init__(self) -> None:
        # Load settings if present
        cfg: Dict[str, Any] = {}
        try:
            with open("settings/config.json") as f:
                data = json.load(f)
            cfg = data.get("brains", {}).get("velocity", {}) if isinstance(data, dict) else {}
        except Exception:
            cfg = {}

        self.BLOCK = int(cfg.get("BLOCK", 12))
        self.FLAT_BAND_DEG = float(cfg.get("FLAT_BAND_DEG", 8.0))
        self.BIG_CHANGE_DEG = float(cfg.get("BIG_CHANGE_DEG", 24.0))
        self.RUN_LEN_MIN = int(cfg.get("RUN_LEN_MIN", 2))
        self.HORIZON = int(cfg.get("HORIZON", 24))
        self.PCT_MOVE = float(cfg.get("PCT_MOVE", 0.01))

        self._pts = {
            "vel_up": {"x": [], "y": [], "s": [], "c": "green"},
            "vel_down": {"x": [], "y": [], "s": [], "c": "red"},
            "vel_flat": {"x": [], "y": [], "s": [], "c": "gray"},
            "vel_reversal": {"x": [], "y": []},
        }
        self._last_angle: float | None = None
        self._last_class: int | None = None
        self._run_len = 0
        self._block_ix = -1
        self._events: List[Dict[str, Any]] = []
        self._bubbles = {"up": [], "down": []}
        self._run_peak = 0.0

    # Interface -----------------------------------------------------------------
    def warmup(self) -> int:
        return self.BLOCK * 2

    def prepare(self, df: pd.DataFrame) -> None:
        # No heavy pre-computation for now
        return None

    def step(self, df: pd.DataFrame, t: int) -> None:
        if t % self.BLOCK != 0:
            return
        self._block_ix += 1

        sub_now = df["close"].iloc[t - self.BLOCK + 1 : t + 1]
        if len(sub_now) > 1:
            slope_now = float(np.polyfit(np.arange(len(sub_now)), sub_now, 1)[0])
        else:
            slope_now = 0.0
        angle_now = float(np.degrees(np.arctan(slope_now)))

        dangle = 0.0 if self._last_angle is None else angle_now - self._last_angle

        if angle_now > self.FLAT_BAND_DEG:
            klass = 1
        elif angle_now < -self.FLAT_BAND_DEG:
            klass = -1
        else:
            klass = 0

        prior_run_len = self._run_len
        if klass == self._last_class and klass != 0:
            self._run_len += 1
        else:
            if self._last_class in (1, -1) and self._run_peak > 0:
                key = "up" if self._last_class == 1 else "down"
                self._bubbles[key].append(self._run_peak)
            self._run_len = 1 if klass != 0 else 0
            self._run_peak = 0.0

        self._run_peak = max(self._run_peak, abs(dangle))

        x = int(df["candle_index"].iloc[t])
        y = float(df["close"].iloc[t])
        size = max(10.0, min(150.0, abs(dangle) * 2.0))
        if klass == 1:
            self._pts["vel_up"]["x"].append(x)
            self._pts["vel_up"]["y"].append(y)
            self._pts["vel_up"]["s"].append(size)
        elif klass == -1:
            self._pts["vel_down"]["x"].append(x)
            self._pts["vel_down"]["y"].append(y)
            self._pts["vel_down"]["s"].append(size)
        else:
            self._pts["vel_flat"]["x"].append(x)
            self._pts["vel_flat"]["y"].append(y)
            self._pts["vel_flat"]["s"].append(size)

        big_rev = False
        if (
            self._last_class in {1, -1}
            and (klass == 0 or klass == -self._last_class)
            and abs(dangle) >= self.BIG_CHANGE_DEG
            and prior_run_len >= self.RUN_LEN_MIN
        ):
            self._pts["vel_reversal"]["x"].append(x)
            self._pts["vel_reversal"]["y"].append(y)
            big_rev = True

        self._events.append(
            {
                "t": t,
                "price": y,
                "tag": "reversal" if big_rev else "",
                "angle": angle_now,
                "dangle": dangle,
                "klass": klass,
                "run_len": prior_run_len,
            }
        )

        self._last_angle = angle_now
        self._last_class = klass

    def overlays(self) -> Dict[str, Dict[str, List[Any]]]:
        return self._pts

    # ===================== Stats =====================
    def compute_stats(
        self, df: pd.DataFrame, _trend_state: List[int], _slopes: List[float]
    ) -> Dict[str, Any]:
        events = self._events
        runs: List[tuple[int, int, int]] = []
        if events:
            cur = events[0]["klass"]
            start = 0
            for i in range(1, len(events)):
                if events[i]["klass"] != cur:
                    runs.append((cur, start, i - 1))
                    cur = events[i]["klass"]
                    start = i
            runs.append((cur, start, len(events) - 1))

        up_lens: List[int] = []
        down_lens: List[int] = []
        for dir_, s, e in runs:
            length = e - s + 1
            if dir_ == 1:
                up_lens.append(length)
            elif dir_ == -1:
                down_lens.append(length)

        avg_up_run = float(np.mean(up_lens)) if up_lens else 0.0
        avg_down_run = float(np.mean(down_lens)) if down_lens else 0.0
        avg_abs_dangle = (
            float(np.mean([abs(ev["dangle"]) for ev in events])) if events else 0.0
        )

        hits = 0
        total = 0
        bars_to_hit: List[int] = []
        for i, ev in enumerate(events):
            if ev["tag"] != "reversal":
                continue
            total += 1
            prior_sign = None
            j = i - 1
            while j >= 0:
                if events[j]["klass"] in (1, -1):
                    prior_sign = events[j]["klass"]
                    break
                j -= 1
            if prior_sign is None:
                continue
            t_event = ev["t"]
            price_event = ev["price"]
            window = df["close"].iloc[t_event + 1 : t_event + 1 + self.HORIZON]
            hit_index = None
            if prior_sign == 1:
                target = price_event * (1 - self.PCT_MOVE)
                for idx, price in enumerate(window, start=1):
                    if price <= target:
                        hit_index = idx
                        break
            else:
                target = price_event * (1 + self.PCT_MOVE)
                for idx, price in enumerate(window, start=1):
                    if price >= target:
                        hit_index = idx
                        break
            if hit_index is not None:
                hits += 1
                bars_to_hit.append(hit_index)
        rate = hits / total if total else 0.0
        avg_bars = float(np.mean(bars_to_hit)) if bars_to_hit else 0.0

        avg_bubble_up = (
            float(np.mean(self._bubbles["up"])) if self._bubbles["up"] else 0.0
        )
        avg_bubble_down = (
            float(np.mean(self._bubbles["down"])) if self._bubbles["down"] else 0.0
        )

        return {
            "avg_up_run_len_blocks": avg_up_run,
            "avg_down_run_len_blocks": avg_down_run,
            "avg_abs_delta_angle": avg_abs_dangle,
            "cyan_success": {
                "hits": hits,
                "total": total,
                "rate": rate,
                "avg_bars_to_hit": avg_bars,
            },
            "avg_bubble_size_up": avg_bubble_up,
            "avg_bubble_size_down": avg_bubble_down,
        }
