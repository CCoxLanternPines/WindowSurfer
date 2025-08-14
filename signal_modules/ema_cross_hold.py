NAME = "ema_cross_hold"
LOOKBACK = 20

def calculate(df, i):
    if i < LOOKBACK:
        return 0
    ema_fast = df["close"].ewm(span=5).mean()
    ema_slow = df["close"].ewm(span=20).mean()
    if ema_fast.iloc[i] > ema_slow.iloc[i] and ema_fast.iloc[i-1] <= ema_slow.iloc[i-1]:
        return +1
    elif ema_fast.iloc[i] < ema_slow.iloc[i] and ema_fast.iloc[i-1] >= ema_slow.iloc[i-1]:
        return -1
    return 0
