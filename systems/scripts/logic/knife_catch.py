from __future__ import annotations

from typing import Optional, List, Dict


def should_buy(candle: Dict, window_data: Dict, active_notes: List[Dict], verbose: bool = False) -> Optional[Dict]:
    """Buy based on falling tunnel position logic."""
    if not window_data:
        return None
    position = window_data.get("tunnel_position", 0)
    existing = next((n for n in active_notes if n.get("strategy") == "knife_catch"), None)

    if verbose:
        print(f"[knife_catch] evaluate buy position={position} has_note={existing is not None}")

    if existing is None:
        if position == 0:
            return {
                "strategy": "knife_catch",
                "price": candle["close"],
                "window": window_data.get("window_label"),
                "entry_tunnel_position": position,
                "entry_window_position": window_data.get("window_position"),
                "timestamp": candle["timestamp"],
            }
        return None

    entry_pos = existing.get("entry_tunnel_position", 0)
    if position < entry_pos:
        return {
            "strategy": "knife_catch",
            "price": candle["close"],
            "window": window_data.get("window_label"),
            "entry_tunnel_position": position,
            "entry_window_position": window_data.get("window_position"),
            "timestamp": candle["timestamp"],
        }

    return None


def should_sell(note: Dict, candle: Dict, window_data: Dict, verbose: bool = False) -> bool:
    """Sell when tunnel and window positions recover."""
    if not window_data:
        return False
    position = window_data.get("tunnel_position", 0)
    window_pos = window_data.get("window_position", 0)

    if verbose:
        print(
            f"[knife_catch] evaluate sell position={position} window_pos={window_pos} entry_window={note.get('entry_window_position')}"
        )

    return position > 0.9 and window_pos >= note.get("entry_window_position", 0)

