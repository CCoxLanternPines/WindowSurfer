from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List


class LedgerManager:
    """Minimal in-memory ledger tracking coin amounts and fiat."""

    def __init__(
        self,
        tag: str | None = None,
        *,
        fiat_start: float = 0.0,
        min_order_usd: float = 0.0,
        allow_partial_buys: bool = True,
    ) -> None:
        self.tag = tag
        self.fiat_available: float = float(fiat_start)
        self.min_order_usd: float = float(min_order_usd)
        self.allow_partial_buys: bool = bool(allow_partial_buys)
        self.open_notes: List[Dict] = []
        self.closed_notes: List[Dict] = []
        self._next_id = 1
        self.realized_usd: float = 0.0

    def buy(self, coin_amount: float, price: float, ts: int) -> Dict:
        requested_coin = float(coin_amount)
        if requested_coin <= 0 or price <= 0:
            return {"requested": requested_coin, "filled": 0.0, "reason": "invalid"}

        affordable_coin = self.fiat_available / price
        if affordable_coin <= 0:
            return {"requested": requested_coin, "filled": 0.0, "reason": "no_fiat"}

        fill_coin = (
            min(requested_coin, affordable_coin)
            if self.allow_partial_buys
            else requested_coin
        )
        if not self.allow_partial_buys and fill_coin < requested_coin:
            return {
                "requested": requested_coin,
                "filled": 0.0,
                "reason": "insufficient_fiat",
            }

        spend_usd = fill_coin * price
        if spend_usd < self.min_order_usd:
            return {
                "requested": requested_coin,
                "filled": 0.0,
                "reason": "below_min_order",
            }

        self.fiat_available -= spend_usd
        note = {
            "id": self._next_id,
            "entry_ts": ts,
            "entry_price": price,
            "coin_amount": fill_coin,
            "remaining": fill_coin,
            "fills": [],
        }
        self._next_id += 1
        self.open_notes.append(note)
        return {
            "requested": requested_coin,
            "filled": fill_coin,
            "spent_usd": spend_usd,
            "note_id": note["id"],
        }

    def sell(
        self, coin_amount: float, price: float, ts: int, allow_loss: bool = True
    ) -> Dict:
        """Sell ``coin_amount`` of coin using highest entry-price notes first."""
        coin_left = coin_amount
        fills: List[Dict] = []
        # Highest entry price first
        self.open_notes.sort(key=lambda n: n["entry_price"], reverse=True)
        proceeds_usd = 0.0
        for note in list(self.open_notes):
            if coin_left <= 0:
                break
            if not allow_loss and price < note["entry_price"]:
                continue
            take = min(note["remaining"], coin_left)
            if take <= 0:
                continue
            fill = {
                "note_id": note["id"],
                "amount": take,
                "sell_price": price,
                "sell_ts": ts,
            }
            note["remaining"] -= take
            note.setdefault("fills", []).append(fill)
            fills.append(fill)
            coin_left -= take
            pnl = (price - note["entry_price"]) * take
            self.realized_usd += pnl
            proceeds_usd += take * price
            if note["remaining"] <= 0:
                self.open_notes.remove(note)
                closed = dict(note)
                self.closed_notes.append(closed)
        self.fiat_available += proceeds_usd
        return {
            "requested": coin_amount,
            "filled": coin_amount - coin_left,
            "fills": fills,
            "proceeds_usd": proceeds_usd,
        }

    # ---- Read helpers -------------------------------------------------
    def get_open_notes(self) -> List[Dict]:
        return list(self.open_notes)

    def get_closed_notes(self) -> List[Dict]:
        return list(self.closed_notes)

    def total_coin(self) -> float:
        return sum(n["remaining"] for n in self.open_notes)

    def realized_gain_usd(self) -> float:
        return self.realized_usd

    def unrealized_gain_usd(self, price: float) -> float:
        return sum(
            (price - n["entry_price"]) * n["remaining"] for n in self.open_notes
        )

    def total_pnl_usd(self, price: float) -> float:
        return self.realized_gain_usd() + self.unrealized_gain_usd(price)

    def value_usd(self, price: float) -> float:
        return self.total_coin() * price

    def summary(self) -> Dict:
        return {
            "open_notes": len(self.open_notes),
            "closed_notes": len(self.closed_notes),
            "total_coin": self.total_coin(),
            "fiat_available": self.fiat_available,
        }

    # ---- I/O helpers --------------------------------------------------
    def save(self, path: Path) -> None:
        data = {
            "fiat_available": self.fiat_available,
            "open_notes": self.get_open_notes(),
            "closed_notes": self.get_closed_notes(),
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    def save_summary(self, path: Path, final_price: float) -> None:
        data = {
            "final_price": final_price,
            "fiat_available": self.fiat_available,
            "realized_usd": self.realized_gain_usd(),
            "unrealized_usd": self.unrealized_gain_usd(final_price),
            "total_pnl_usd": self.total_pnl_usd(final_price),
            "remaining_coin": self.total_coin(),
            "open_notes": len(self.open_notes),
            "closed_notes": len(self.closed_notes),
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    def get_fiat(self) -> float:
        return self.fiat_available
