def evaluate_buy(candle: dict, window_data: dict, verbose: bool = False) -> bool:
    """
    Debug stub for buy logic.
    Prints tunnel and window position.
    Returns False always (no actual logic yet).
    """
    tunnel_pos = window_data.get("tunnel_position")
    window_pos = window_data.get("window_position")
    
    if verbose:
        print(
            f"[EVAL BUY] tunnel_position={tunnel_pos:.4f} | window_position={window_pos:.4f}"
        )
    
    # TODO: plug in actual knife/whale/fish logic
    return False
