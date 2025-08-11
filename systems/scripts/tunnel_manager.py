from __future__ import annotations

from typing import Dict, List
from datetime import datetime

import logging

from .ledger import Ledger
from .tunnel import Tunnel
from .logger import logger


class TunnelManager:
    """Runs a collection of tunnels sharing a common ledger and capital."""

    def __init__(self, cfg: Dict, capital: float) -> None:
        self.ledger = Ledger(cfg.get("ledger", {}))
        self.capital = capital
        self.tunnels: Dict[str, Dict[str, Tunnel]] = {}

        tunnels_cfg = cfg.get("tunnels", {})
        for symbol, tunnel_map in tunnels_cfg.items():
            self.tunnels[symbol] = {
                tunnel_id: Tunnel(tcfg, symbol, tunnel_id) for tunnel_id, tcfg in tunnel_map.items()
            }

    # ------------------------------------------------------------------
    def tick(self, prices: Dict[str, float], timestamp: datetime) -> Dict[str, List[str]]:
        """Process one tick of prices. Returns logs of actions."""
        actions: Dict[str, List[str]] = {sym: [] for sym in prices.keys()}

        # Update windows first with optional summaries
        for symbol, price in prices.items():
            for tunnel_id, tunnel in self.tunnels.get(symbol, {}).items():
                tunnel.update_window(price)
                if logger.isEnabledFor(logging.INFO):
                    pos = tunnel.current_position
                    pos_str = f"{pos:.3f}" if pos is not None else "nan"
                    logger.info(
                        f"[TICK] {symbol}/{tunnel_id} pos={pos_str} buy_trig={tunnel.buy_trigger_position:.3f} "
                        f"maturity_mult={tunnel.sell_maturity_multiplier:.3f} can_buy={tunnel.can_buy}"
                    )

        debug_mode = logger.isEnabledFor(logging.DEBUG)

        # Handle sells first to free capital
        for symbol, price in prices.items():
            for tunnel_id, tunnel in self.tunnels.get(symbol, {}).items():
                notes = self.ledger.get_notes(symbol, tunnel_id)
                sells = tunnel.check_sell_opportunities(notes, price, debug=debug_mode)
                for sell in sells:
                    fiat = self.ledger.sell(
                        symbol,
                        tunnel_id,
                        sell["qty"],
                        price,
                        note_idx=sell["note_idx"],
                        partial=sell.get("partial", False),
                    )
                    if fiat > 0:
                        self.capital += fiat
                        actions[symbol].append(
                            f"sell {tunnel_id} {sell['qty']:.4f} @ {price:.2f}"
                        )
                        logger.warning(
                            f"[SELL] {symbol}/{tunnel_id} sold {sell['qty']:.4f} @ {price:.2f}"
                        )

        # Compute available funds
        open_value = self.ledger.total_fiat_value(prices)
        available = self.capital - open_value

        # Gather buy orders
        buy_orders = []
        for symbol, price in prices.items():
            for tunnel_id, tunnel in self.tunnels.get(symbol, {}).items():
                qty = tunnel.check_buy_opportunity(price, debug=debug_mode)
                if qty > 0:
                    cost = qty * price
                    maturity_price = price * tunnel.sell_maturity_multiplier
                    buy_orders.append(
                        {
                            "symbol": symbol,
                            "tunnel_id": tunnel_id,
                            "qty": qty,
                            "price": price,
                            "cost": cost,
                            "maturity_price": maturity_price,
                        }
                    )

        # Execute buys respecting capital
        for order in buy_orders:
            if order["cost"] > available:
                continue
            ok = self.ledger.buy(
                order["symbol"],
                order["tunnel_id"],
                order["qty"],
                order["price"],
                order["maturity_price"],
                timestamp=timestamp,
            )
            if ok:
                available -= order["cost"]
                actions[order["symbol"]].append(
                    f"buy {order['tunnel_id']} {order['qty']:.4f} @ {order['price']:.2f}"
                )
                logger.warning(
                    f"[BUY] {order['symbol']}/{order['tunnel_id']} bought {order['qty']:.4f} @ {order['price']:.2f}"
                )
        return actions
