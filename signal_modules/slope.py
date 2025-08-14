import numpy as np

NAME = "slope"
LOOKBACK = 10

def calculate(df, i):
    if i < LOOKBACK:
        return 0
    closes = df["close"].iloc[i-LOOKBACK+1:i+1].values
    x = np.arange(len(closes))
    slope = np.polyfit(x, closes, 1)[0]
    if slope > 0:
        return +1
    elif slope < 0:
        return -1
    return 0
