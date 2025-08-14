NAME = "ma_diff"
LOOKBACK = 10

def calculate(df, i):
    short_ma = df["close"].rolling(3).mean()
    long_ma = df["close"].rolling(10).mean()
    if i < LOOKBACK:
        return 0
    if short_ma.iloc[i] > long_ma.iloc[i]:
        return +1
    elif short_ma.iloc[i] < long_ma.iloc[i]:
        return -1
    return 0
