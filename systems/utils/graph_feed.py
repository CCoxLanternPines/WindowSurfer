from __future__ import annotations

"""Lightweight NDJSON feed writer for graphing."""

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional


class GraphFeed:
    """Append-only JSON line feed used by both simulation and live runs."""

    def __init__(
        self,
        *,
        mode: str,
        coin: str,
        account: Optional[str] = None,
        sim_dir: str = "data/temp/simulation",
        live_dir: str = "data/temp",
        downsample: int = 1,
        flush: bool = False,
        run_ts: Optional[str] = None,
    ) -> None:
        mode = mode.lower()
        if mode not in ("sim", "live"):
            raise ValueError("mode must be 'sim' or 'live'")

        self.mode = mode
        self.coin = coin.replace("/", "").upper()
        self.account = account
        self.downsample = max(1, int(downsample))
        self.flush = bool(flush)
        ts = run_ts or datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

        if mode == "sim":
            base = Path(sim_dir)
            base.mkdir(parents=True, exist_ok=True)
            self.path = base / f"{self.coin}_{ts}.json"
            file_mode = "w"
        else:
            if not account:
                raise ValueError("account required for live mode")
            base = Path(live_dir)
            base.mkdir(parents=True, exist_ok=True)
            self.path = base / f"{account}_{self.coin}.json"
            file_mode = "a"

        self._fh = self.path.open(file_mode, encoding="utf-8")
        meta = {"t": "meta", "v": 1, "mode": mode, "coin": self.coin, "account": account, "tz": "UTC"}
        self._write(meta)

    # ------------------------------------------------------------------
    def _write(self, obj: dict) -> None:
        line = json.dumps(obj, separators=(",", ":"), ensure_ascii=False)
        self._fh.write(line + "\n")
        if self.flush:
            self._fh.flush()

    # ------------------------------------------------------------------
    def candle(self, i: int, ts: Optional[int], o: float, h: float, l: float, c: float) -> None:
        if i % self.downsample != 0:
            return
        payload = {"t": "c", "i": i, "ts": ts, "o": o, "h": h, "l": l, "c": c}
        self._write(payload)

    def indicator(self, i: int, k: str, v: float) -> None:
        if i % self.downsample != 0:
            return
        payload = {"t": "ind", "i": i, "k": k, "v": v}
        self._write(payload)

    def buy(self, i: int, p: float, u: float, usd: float, target: float) -> None:
        payload = {"t": "buy", "i": i, "p": p, "u": u, "$": usd, "target": target}
        self._write(payload)

    def sell(self, i: int, p: float, u: float, usd: float, entry: float) -> None:
        payload = {"t": "sell", "i": i, "p": p, "u": u, "$": usd, "entry": entry}
        self._write(payload)

    def capital(self, i: int, free: float, equity: float) -> None:
        payload = {"t": "cap", "i": i, "free": free, "equity": equity}
        self._write(payload)

    def close(self) -> None:
        try:
            self._fh.flush()
            self._fh.close()
        except Exception:
            pass
