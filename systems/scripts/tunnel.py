from __future__ import annotations

from collections import deque
from typing import Dict, List, Optional

import logging

from .logger import logger


class Tunnel:
    """Maintains window based state and trading triggers."""

    def __init__(self, cfg: Dict, symbol: str, tunnel_id: str) -> None:
        self.cfg = cfg
        self.symbol = symbol
        self.tunnel_id = tunnel_id
        self.window_size_hours = int(cfg.get("window_size_hours", 24))
        self.base_bet_fraction = float(cfg.get("base_bet_fraction", 0.0))
        self.buy_trigger_position = float(cfg.get("buy_trigger_position", 0.0))
        self.rebuy_unlock_position = float(cfg.get("rebuy_unlock_position", 1.0))
        self.sell_maturity_multiplier = float(cfg.get("sell_maturity_multiplier", 1.0))
        self.min_roi = float(cfg.get("min_roi", 0.0))
        self.partial_sell_midpoint = float(cfg.get("partial_sell_midpoint", 0.5))
        self.wtf = cfg.get("wtf", {
            "enabled": False,
            "min": 1.0,
            "max": 1.0,
        })

        self.prices: deque = deque(maxlen=self.window_size_hours)
        self.current_position: Optional[float] = None
        self.window_high: Optional[float] = None
        self.window_low: Optional[float] = None
        self.max_position_ever: float = 0.0
        self.lowest_mix_position_ever: float = 1.0

        self.can_buy = True
        self.buy_reset_triggered = False

    # ------------------------------------------------------------------
    def update_window(self, price: float) -> None:
        self.prices.append(price)
        if len(self.prices) == 0:
            return
        self.window_high = max(self.prices)
        self.window_low = min(self.prices)
        if self.window_high == self.window_low:
            self.current_position = None
            return
        self.current_position = (price - self.window_low) / (self.window_high - self.window_low)
        self.max_position_ever = max(self.max_position_ever, self.current_position)
        self.lowest_mix_position_ever = min(self.lowest_mix_position_ever, self.current_position)

        # Lockout reset handling
        if not self.can_buy and not self.buy_reset_triggered:
            if self.current_position >= self.rebuy_unlock_position:
                self.buy_reset_triggered = True
        if not self.can_buy and self.buy_reset_triggered:
            if self.current_position <= self.buy_trigger_position:
                self.can_buy = True
                self.buy_reset_triggered = False

    # ------------------------------------------------------------------
    def calc_wtf_multiplier(self) -> float:
        if not self.wtf.get("enabled", False) or self.current_position is None:
            return 1.0
        # Basic example formula using inverse position
        mult = 1 + (1 - self.current_position)
        wtf_min = self.wtf.get("min", 1.0)
        wtf_max = self.wtf.get("max", 1.0)
        return max(wtf_min, min(wtf_max, mult))

    # ------------------------------------------------------------------
    def check_buy_opportunity(self, price: float, debug: bool = False) -> float:
        if debug:
            logger.debug(
                f"[BUYCHK] {self.symbol}/{self.tunnel_id} pos={self.current_position:.3f} "
                f"buy_trig={self.buy_trigger_position:.3f} can_buy={self.can_buy} "
                f"reset={self.buy_reset_triggered} wtf_mult={self.calc_wtf_multiplier():.2f}"
            )
        if self.current_position is None:
            return 0.0
        if self.can_buy and self.current_position <= self.buy_trigger_position:
            qty = self.base_bet_fraction * self.calc_wtf_multiplier()
            self.can_buy = False
            return qty
        if debug and not self.can_buy:
            logger.debug(
                f"[BUYCHK] Skipped — cooldown active for {self.symbol}/{self.tunnel_id}"
            )
        return 0.0

    # ------------------------------------------------------------------
    def check_sell_opportunities(self, notes: List, price: float, debug: bool = False) -> List[Dict]:
        if self.current_position is None:
            return []
        if debug:
            logger.debug(
                f"[SELLCHK] {self.symbol}/{self.tunnel_id} pos={self.current_position:.3f} "
                f"maturity_mult={self.sell_maturity_multiplier:.3f} min_roi={self.min_roi:.3f}"
            )
        sells: List[Dict] = []
        for idx, note in enumerate(notes):
            # Partial sell at midpoint
            if (
                self.partial_sell_midpoint is not None
                and not note.partial_sold
            ):
                midpoint = note.buy_price + (
                    note.maturity_price - note.buy_price
                ) * self.partial_sell_midpoint
                if price >= midpoint:
                    sells.append({"note_idx": idx, "qty": note.qty / 2, "partial": True})
                    continue
            maturity_price = note.maturity_price * self.sell_maturity_multiplier
            roi = (price - note.buy_price) / note.buy_price
            if price >= maturity_price and roi >= self.min_roi:
                sells.append({"note_idx": idx, "qty": note.qty, "partial": False})
        return sells
