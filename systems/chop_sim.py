import logging
from dataclasses import dataclass
from typing import List

from .io_candles import load_candles
from .scripts.chop_policy import PolicyState, compute_features, should_buy, should_sell


@dataclass
class Position:
    entry: float
    qty: float
    bars_held: int = 0


def run(args):
    candles = load_candles(args.tag, args.path_csv, args.allow_gaps)
    state = PolicyState(window=args.window, slope_gap=args.slope_gap)

    cash = args.start_capital
    equity = cash
    positions: List[Position] = []
    trades = []
    buy_cd = 0
    sell_cd = 0
    equity_curve = []
    logger = logging.getLogger('chop')

    for dt, candle in candles:
        features = compute_features(state, candle)
        if buy_cd > 0:
            buy_cd -= 1
        if sell_cd > 0:
            sell_cd -= 1

        if features['ready']:
            ctx = {
                'chop_slope_eps': args.chop_slope_eps,
                'atr_min': args.atr_min,
                'atr_max': args.atr_max,
                'entry_dev_pct': args.entry_dev_pct,
                'buy_cooldown': buy_cd,
                'position_count': len(positions),
                'max_concurrent': args.max_concurrent,
            }
            decision, reason = should_buy(features, ctx)
            if decision:
                qty = (equity * args.risk_frac) / features['close']
                positions.append(Position(entry=features['close'], qty=qty))
                cash -= qty * features['close']
                buy_cd = args.buy_cooldown
                if args.verbose >= 1:
                    logger.info(f'BUY {qty:.6f} @ {features["close"]:.2f}')
                if args.verbose >= 2:
                    logger.info(f'Features: {features}')
            elif args.verbose >= 3:
                logger.debug(f'BUY blocked: {reason}')

        for position in list(positions):
            ctx_sell = {
                'take_profit_pct': args.take_profit_pct,
                'stop_loss_pct': args.stop_loss_pct,
                'max_hold_bars': args.max_hold_bars,
                'sell_cooldown': sell_cd,
            }
            decision, reason = should_sell(position.__dict__, features, ctx_sell)
            if decision:
                if sell_cd == 0:
                    cash += position.qty * features['close']
                    roi = (features['close'] - position.entry) / position.entry
                    trades.append({'roi': roi, 'bars': position.bars_held})
                    positions.remove(position)
                    sell_cd = args.sell_cooldown
                    if args.verbose >= 1:
                        logger.info(f'SELL {position.qty:.6f} @ {features["close"]:.2f} roi {roi*100:.2f}%')
                    if args.verbose >= 2:
                        logger.info(f'Features: {features}')
                else:
                    if args.verbose >= 3:
                        logger.debug('SELL blocked: SELL_COOLDOWN')
            else:
                if args.verbose >= 3 and reason != 'HOLD':
                    logger.debug(f'SELL blocked: {reason}')
            position.bars_held += 1

        equity = cash + sum(p.qty * features['close'] for p in positions)
        equity_curve.append(equity)

    final_pct = (equity / args.start_capital - 1) * 100
    buy_hold_pct = (candles[-1][1]['close'] / candles[0][1]['close'] - 1) * 100

    wins = [t for t in trades if t['roi'] > 0]
    losses = [t for t in trades if t['roi'] <= 0]
    win_rate = (len(wins) / len(trades) * 100) if trades else 0.0
    avg_win = sum(t['roi'] for t in wins) / len(wins) * 100 if wins else 0.0
    avg_loss = sum(t['roi'] for t in losses) / len(losses) * 100 if losses else 0.0
    avg_hold = sum(t['bars'] for t in trades) / len(trades) if trades else 0.0

    print(f'Chop PnL: {final_pct:.2f}%')
    print(f'Buy & Hold: {buy_hold_pct:.2f}%')
    print('Trades:', len(trades))
    print(f'Win rate: {win_rate:.2f}%')
    print(f'Avg win %: {avg_win:.2f}%')
    print(f'Avg loss %: {avg_loss:.2f}%')
    print(f'Avg hold bars: {avg_hold:.2f}')

    if args.report_extended and equity_curve:
        max_equity = equity_curve[0]
        max_dd = 0
        for eq in equity_curve:
            if eq > max_equity:
                max_equity = eq
            dd = (max_equity - eq) / max_equity
            if dd > max_dd:
                max_dd = dd
        span_years = (candles[-1][0] - candles[0][0]).total_seconds() / (365*24*3600)
        cagr = (equity / args.start_capital) ** (1 / span_years) - 1 if span_years > 0 else 0
        mar = cagr / max_dd if max_dd > 0 else 0
        print(f'Max DD: {max_dd*100:.2f}%')
        print(f'MAR: {mar:.2f}')
