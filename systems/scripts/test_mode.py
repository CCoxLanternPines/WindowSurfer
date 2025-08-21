from __future__ import annotations

"""Micro trade test mode."""

from systems.scripts.trade_apply import paper_execute_buy, paper_execute_sell


def run_test_mode(account: str, market: str) -> None:
    """Execute a tiny buy and sell to verify pipeline."""
    try:
        buy_res = paper_execute_buy(price=1.0, amount_usd=5.0)
        sell_res = paper_execute_sell(price=1.0, coin_amount=buy_res.get("filled_amount", 0.0))
    except Exception as exc:  # pragma: no cover - simple error display
        print(f"[TEST][FAIL] {exc}")
        return

    if buy_res and sell_res:
        print(f"[TEST][PASS] Account={account} Market={market}")
        print(
            f"Buy executed @{buy_res.get('avg_price')}, Sell executed @{sell_res.get('avg_price')}"
        )
    else:
        print(f"[TEST][FAIL] Account={account} Market={market}")
