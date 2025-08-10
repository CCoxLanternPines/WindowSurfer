from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, Tuple


@dataclass
class PolicyState:
    window: int
    slope_gap: int
    close: Deque[float] = field(default_factory=deque)
    high: Deque[float] = field(default_factory=deque)
    low: Deque[float] = field(default_factory=deque)
    ma_history: Deque[float] = field(default_factory=deque)
    tr: Deque[float] = field(default_factory=deque)
    sum_close: float = 0.0
    sum_tr: float = 0.0
    last_close: float = None


def compute_features(state: PolicyState, candle: Dict[str, float]) -> Dict[str, float]:
    close = float(candle['close'])
    high = float(candle['high'])
    low = float(candle['low'])

    state.close.append(close)
    state.high.append(high)
    state.low.append(low)
    state.sum_close += close
    if len(state.close) > state.window:
        state.sum_close -= state.close.popleft()
        state.high.popleft()
        state.low.popleft()

    ma = state.sum_close / len(state.close)
    state.ma_history.append(ma)
    if len(state.ma_history) > state.window + state.slope_gap:
        state.ma_history.popleft()

    if state.last_close is None:
        tr = high - low
    else:
        tr = max(high - low, abs(high - state.last_close), abs(low - state.last_close))
    state.tr.append(tr)
    state.sum_tr += tr
    if len(state.tr) > state.window:
        state.sum_tr -= state.tr.popleft()

    state.last_close = close

    ready = len(state.close) == state.window and len(state.ma_history) > state.slope_gap and len(state.tr) == state.window

    atr = state.sum_tr / len(state.tr) if state.tr else 0.0
    slope = 0.0
    if len(state.ma_history) > state.slope_gap:
        ma_prev = state.ma_history[-1 - state.slope_gap]
        if ma != 0:
            slope = (ma - ma_prev) / (state.slope_gap * ma)
    dev_pct = (close - ma) / ma if ma else 0.0
    atr_pct = atr / ma if ma else 0.0

    return {
        'ready': ready,
        'close': close,
        'ma': ma,
        'slope': slope,
        'dev_pct': dev_pct,
        'atr_pct': atr_pct,
    }


def should_buy(features: Dict[str, float], ctx: Dict[str, float]) -> Tuple[bool, str]:
    if not features.get('ready'):
        return False, 'NOT_READY'
    if abs(features['slope']) > ctx['chop_slope_eps']:
        return False, 'SLOPE_OUTSIDE'
    if not (ctx['atr_min'] <= features['atr_pct'] <= ctx['atr_max']):
        return False, 'ATR_OUTSIDE'
    if features['dev_pct'] > -ctx['entry_dev_pct']:
        return False, 'DEV_TOO_HIGH'
    if ctx['buy_cooldown'] > 0:
        return False, 'BUY_COOLDOWN'
    if ctx['position_count'] >= ctx['max_concurrent']:
        return False, 'MAX_CONCURRENT'
    return True, 'OK'


def should_sell(position: Dict[str, float], features: Dict[str, float], ctx: Dict[str, float]) -> Tuple[bool, str]:
    if ctx['sell_cooldown'] > 0:
        return False, 'SELL_COOLDOWN'
    change = (features['close'] - position['entry']) / position['entry']
    if change >= ctx['take_profit_pct']:
        return True, 'TP'
    if change <= -ctx['stop_loss_pct']:
        return True, 'SL'
    if position['bars_held'] >= ctx['max_hold_bars']:
        return True, 'TIMEOUT'
    return False, 'HOLD'
