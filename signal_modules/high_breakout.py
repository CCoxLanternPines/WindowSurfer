NAME = "high_breakout"
LOOKBACK = 20

def calculate(df, i):
    if i < LOOKBACK:
        return 0
    high_window = df["high"].iloc[i-LOOKBACK:i].max()
    low_window = df["low"].iloc[i-LOOKBACK:i].min()
    if df["close"].iloc[i] > high_window:
        return +1
    elif df["close"].iloc[i] < low_window:
        return -1
    return 0
