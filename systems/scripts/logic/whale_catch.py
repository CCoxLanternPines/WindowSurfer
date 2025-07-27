from __future__ import annotations

from typing import Optional, List, Dict


def should_buy(candle: Dict, window_data: Dict, active_notes: List[Dict], verbose: bool = False) -> Optional[Dict]:
    """Buy when ``tunnel_position`` < 0.1."""
    if not window_data:
        return None
    position = window_data.get("tunnel_position", 0)

    if verbose:
        print(f"[whale_catch] evaluate buy position={position}")

    if position < 0.1:
        note = {
            "strategy": "whale_catch",
            "price": candle["close"],
            "window": window_data.get("window_label"),
            "entry_tunnel_position": position,
            "entry_window_position": window_data.get("window_position"),
            "timestamp": candle["timestamp"],
        }
        return note
    return None


def should_sell(note: Dict, candle: Dict, window_data: Dict, verbose: bool = False) -> bool:
    """Sell when ``tunnel_position`` > 0.9."""
    if not window_data:
        return False
    position = window_data.get("tunnel_position", 0)

    if verbose:
        print(f"[whale_catch] evaluate sell position={position}")

    return position > 0.9
