from __future__ import annotations

"""Jackpot accumulator for simulation mode."""

from dataclasses import dataclass
from typing import Optional

from systems.utils.addlog import addlog


@dataclass
class BuyQuote:
    amount_quote: float


@dataclass
class ExitAll:
    pass


class JackpotManager:
    """Manage profit-funded jackpot DCA buying and exiting."""

    def __init__(self, cfg: dict, start_ts: int, ath_from_data: float, min_from_data: float) -> None:
        self.cfg = cfg
        interval = int(cfg.get("drip_interval_hours", 0) * 3600)
        self.last_drip_ts = start_ts - interval
        self.bank = 0.0
        self.bank_in = 0.0
        self.bank_spent = 0.0
        self.held_qty = 0.0
        self.avg_cost = 0.0
        self.ath = ath_from_data
        self.min_price = min_from_data
        self.exit_value: Optional[float] = None
        self.realized_from_exit: Optional[float] = None

    def on_tick(
        self,
        ts: int,
        price: float,
        realized_pnl_delta: float,
        wallet_quote_available: float,
        wallet_total_quote: float,
    ) -> Optional[object]:
        if realized_pnl_delta > 0:
            self.bank += realized_pnl_delta
            self.bank_in += realized_pnl_delta

        exit_frac = self.cfg.get("exit_ath_frac", 0.75)
        if self.held_qty > 0 and price >= exit_frac * self.ath:
            value = self.held_qty * price
            realized = (price - self.avg_cost) * self.held_qty
            addlog(
                f"[JACKPOT][EXIT] ts={ts} price=${price:.4f} qty={self.held_qty:.6f} value=${value:.2f}",
                verbose_int=1,
                verbose_state=True,
            )
            self.exit_value = value
            self.realized_from_exit = realized
            self.held_qty = 0.0
            self.avg_cost = 0.0
            self.last_drip_ts = ts
            return ExitAll()

        dd = 1.0 - price / self.ath if self.ath else 0.0
        dd_start = self.cfg.get("dd_start", 0.5)
        if dd < dd_start:
            return None
        if ts - self.last_drip_ts < int(self.cfg.get("drip_interval_hours", 0) * 3600):
            addlog("[JACKPOT][SKIP] reason=interval", verbose_int=2, verbose_state=True)
            return None
        if self.bank <= 0:
            addlog("[JACKPOT][SKIP] reason=bank", verbose_int=2, verbose_state=True)
            return None

        dd_bottom = self.cfg.get("dd_bottom", 0.85)
        if dd_bottom > dd_start:
            boost = 1.0 + (dd - dd_start) * (2.0 - 1.0) / (dd_bottom - dd_start)
        else:
            boost = 1.0
        boost = max(1.0, min(2.0, boost))
        q = self.bank * self.cfg.get("drip_base_frac_of_bank", 0.0) * boost
        q = min(q, self.bank)
        min_order = self.cfg.get("min_order_quote", 0.0)
        if q < min_order:
            addlog("[JACKPOT][SKIP] reason=min", verbose_int=2, verbose_state=True)
            return None
        market_value = self.held_qty * price
        cap = self.cfg.get("max_jackpot_value_frac", 1.0) * wallet_total_quote
        if market_value + q > cap:
            addlog("[JACKPOT][SKIP] reason=cap", verbose_int=2, verbose_state=True)
            return None

        qty = q / price if price else 0.0
        total_cost = self.avg_cost * self.held_qty + q
        self.held_qty += qty
        self.avg_cost = total_cost / self.held_qty if self.held_qty else 0.0
        self.bank -= q
        self.bank_spent += q
        self.last_drip_ts = ts
        addlog(
            f"[JACKPOT][DRIP] ts={ts} price=${price:.4f} spend=${q:.2f} bank=${self.bank:.2f} dd={dd:.2f} boost={boost:.2f} qty={qty:.6f}",
            verbose_int=1,
            verbose_state=True,
        )
        return BuyQuote(q)

    def snapshot(self) -> dict:
        return {
            "bank_in": self.bank_in,
            "bank_spent": self.bank_spent,
            "qty": self.held_qty,
            "avg_cost": self.avg_cost,
            "exit_value": self.exit_value,
            "realized_from_exit": self.realized_from_exit,
        }
