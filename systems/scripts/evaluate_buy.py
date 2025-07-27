def evaluate_buy(candle: dict, window_data: dict, verbose: bool = False) -> bool:
    if not window_data:
        return False

    tunnel_pos = window_data.get("tunnel_position", 0)
    window_pos = window_data.get("window_position", 0)
    tunnel_high = window_data.get("window_ceiling", 0)
    tunnel_low = window_data.get("window_floor", 0)
    tunnel_height = tunnel_high - tunnel_low
    tunnel_pct = tunnel_pos * 100

    if verbose:
        print(
            f"🧠 Tunnel {{w={tunnel_low:.4f}, h={tunnel_height:.4f}, p={tunnel_pos:.4f}, t={tunnel_pct:.1f}%}} "
            f"Window {{p={window_pos:.4f}}}"
        )

    return False
