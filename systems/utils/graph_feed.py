from __future__ import annotations
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

class GraphFeed:
    def __init__(self, *, mode: str, coin: str, account: Optional[str] = None,
                 sim_dir: str = "data/temp/simulation", live_dir: str = "data/temp",
                 downsample: int = 1, flush: bool = False):
        self.mode = mode
        self.coin = coin.replace("/", "").upper()
        self.account = account
        self.downsample = max(1, int(downsample))
        self.flush = bool(flush)
        self._candle_count = 0
        self._fh = None

        if mode == "sim":
            Path(sim_dir).mkdir(parents=True, exist_ok=True)
            ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
            self.path = Path(sim_dir) / f"{self.coin}_{ts}.json"
        else:  # live
            if not account:
                raise ValueError("account required for live mode")
            Path(live_dir).mkdir(parents=True, exist_ok=True)
            self.path = Path(live_dir) / f"{account}_{self.coin}.json"

        self._fh = self.path.open("a", encoding="utf-8")
        self._write({"t": "meta", "v": 1, "mode": mode, "coin": self.coin,
                     "account": account, "tz": "UTC"})

    def _write(self, obj: dict) -> None:
        self._fh.write(json.dumps(obj, separators=(",", ":")) + "\n")
        if self.flush:
            self._fh.flush()

    def candle(self, i: int, ts: Optional[int], o: float, h: float, l: float, c: float) -> None:
        self._candle_count += 1
        if self._candle_count % self.downsample != 0:
            return
        self._write({"t":"c","i":int(i),"ts":(int(ts) if ts is not None else None),
                     "o":float(o),"h":float(h),"l":float(l),"c":float(c)})

    def indicator(self, i: int, k: str, v: float) -> None:
        # Downsample indicators same as candles
        if self._candle_count % self.downsample != 0:
            return
        self._write({"t":"ind","i":int(i),"k":str(k),"v":float(v)})

    def pressure_bubble(self, i: int, p: float, s: float) -> None:
        """Write a green pressure bubble at candle ``i``."""
        self._write({"t": "pb", "i": int(i), "p": float(p), "s": float(s)})

    def vol_bubble(self, i: int, p: float, s: float) -> None:
        """Write a red volatility bubble at candle ``i``."""
        self._write({"t": "vb", "i": int(i), "p": float(p), "s": float(s)})

    def buy(self, i: int, p: float, u: float, usd: float, target: float) -> None:
        self._write({"t":"buy","i":int(i),"p":float(p),"u":float(u),"$":float(usd),"target":float(target)})

    def sell(self, i: int, p: float, u: float, usd: float, entry: float) -> None:
        self._write({"t":"sell","i":int(i),"p":float(p),"u":float(u),"$":float(usd),"entry":float(entry)})

    def capital(self, i: int, free: float, equity: float) -> None:
        self._write({"t":"cap","i":int(i),"free":float(free),"equity":float(equity)})

    def close(self) -> None:
        try:
            if self._fh:
                self._fh.flush()
                self._fh.close()
        finally:
            self._fh = None
