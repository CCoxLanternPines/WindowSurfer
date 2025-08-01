from __future__ import annotations

"""Execute trading logic at the top of each hour."""

from datetime import datetime
from typing import Any, Dict

from systems.scripts.evaluate_buy import evaluate_buy
from systems.scripts.evaluate_sell import evaluate_sell
from systems.scripts.get_window_data import get_wave_window_data_df
from systems.scripts.kraken_utils import get_live_price
from systems.scripts.ledger import Ledger
from systems.utils.logger import addlog


def handle_top_of_hour(
    *,
    tick: int | datetime,
    sim: bool,
    settings: dict | None = None,
    candle: dict | None = None,
    ledger: Ledger | None = None,
    ledger_config: dict | None = None,
    **kwargs: Any,
) -> None:
    """Run buy/sell evaluations for all windows on an hourly boundary.

    Parameters
    ----------
    tick:
        Current tick index.
    candle:
        Candle data for this tick.
    ledger:
        Ledger tracking open and closed notes.
    ledger_config:
        Ledger-specific configuration.
    sim:
        ``True`` when running in simulation mode. Live logic is not yet
        implemented.
    **kwargs:
        Additional context. The simulation engine passes a DataFrame ``df`` and
        ``offset`` for window calculations along with a mutable ``state``
        dictionary containing capital and cooldown counters.
    """

    if not sim:
        if settings is None:
            return

        for ledger_name, ledger_cfg in settings.get("ledger_settings", {}).items():
            tag = ledger_cfg.get("tag")
            kraken_name = ledger_cfg.get("kraken_name")
            wallet_code = ledger_cfg.get("wallet_code")
            fiat = ledger_cfg.get("fiat")
            window_settings = ledger_cfg.get("window_settings", {})

            ledger = Ledger.load_ledger(tag=ledger_cfg["tag"])

            price = get_live_price(kraken_pair=kraken_name)

            for window_name, window_cfg in window_settings.items():
                if sim:
                    pass  # use sim logic
                else:
                    # live trading logic comes next
                    pass

                summary = ledger.get_account_summary(price)
                print(f"[LIVE] {tag} | {window_name} window")
                print(
                    f"✅ Buy attempts: 0 | Sells: 0 | Open Notes: {summary['open_notes']} | Realized Gain: ${summary['realized_gain']:.2f}"
                )

            Ledger.save_ledger(tag=ledger_cfg["tag"], ledger=ledger)
        return

    if candle is None or ledger is None or ledger_config is None:
        return

    # Extract common context
    df = kwargs.get("df")
    offset = kwargs.get("offset")
    state: Dict[str, Any] = kwargs.get("state", {})
    verbose = kwargs.get("verbose", 0)

    windows = ledger_config.get("window_settings", {})
    if not windows or df is None or offset is None:
        return

    price = float(candle.get("close", 0.0))

    for name, cfg in windows.items():
        wave = get_wave_window_data_df(
            df,
            window=cfg["window_size"],
            candle_offset=offset,
        )
        if not wave:
            continue

        sim_capital = state.get("capital", 0.0)
        max_note_usdt = kwargs.get("max_note_usdt", sim_capital)
        min_note_usdt = kwargs.get("min_note_usdt", 0.0)

        sim_capital, buy_skipped = evaluate_buy(
            ledger=ledger,
            name=name,
            cfg=cfg,
            wave=wave,
            tick=tick,
            price=price,
            sim_capital=sim_capital,
            last_buy_tick=state.get("last_buy_tick", {}),
            max_note_usdt=max_note_usdt,
            min_note_usdt=min_note_usdt,
            verbose=verbose,
        )
        state["capital"] = sim_capital
        if buy_skipped:
            state.get("buy_cooldown_skips", {}).setdefault(name, 0)
            state["buy_cooldown_skips"][name] += 1

        last_sell_tick = state.get("last_sell_tick", {})
        if tick - last_sell_tick.get(name, float("-inf")) < cfg.get("sell_cooldown", 0):
            state.get("sell_cooldown_skips", {}).setdefault(name, 0)
            state["sell_cooldown_skips"][name] += 1
            continue

        sim_capital, closed, roi_skipped = evaluate_sell(
            ledger=ledger,
            name=name,
            tick=tick,
            price=price,
            cfg=cfg,
            sim_capital=sim_capital,
            verbose=verbose,
        )
        state["capital"] = sim_capital
        state["min_roi_gate_hits"] = state.get("min_roi_gate_hits", 0) + roi_skipped

        if closed:
            last_sell_tick[name] = tick
            for note in closed:
                addlog(
                    (
                        f"[SELL] Tick {tick} | Window: {note['window']} | "
                        f"Gain: +${note['gain']:.2f} ({note['gain_pct']:.2%})"
                    ),
                    verbose_int=2,
                    verbose_state=verbose,
                )

    if sim:
        # Simulation-specific behaviour already covered through state mutation.
        # Placeholder for future live logic.
        pass

