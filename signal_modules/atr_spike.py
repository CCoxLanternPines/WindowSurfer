import pandas as pd

NAME = "atr_spike"
LOOKBACK = 20

def calculate(df, i):
    if i < LOOKBACK:
        return 0
    tr = pd.concat([
        df["high"] - df["low"],
        (df["high"] - df["close"].shift()).abs(),
        (df["low"] - df["close"].shift()).abs(),
    ], axis=1).max(axis=1)
    atr = tr.rolling(14).mean()
    if atr.iloc[i] > atr.rolling(50).mean().iloc[i] * 1.5:
        return +1 if df["close"].iloc[i] > df["open"].iloc[i] else -1
    return 0
