import numpy as np

NAME = "angle"
LOOKBACK = 10

def calculate(df, i):
    if i < LOOKBACK:
        return 0
    closes = df["close"].iloc[i-LOOKBACK+1:i+1].values
    x = np.arange(len(closes))
    slope = np.polyfit(x, closes, 1)[0]
    angle = np.degrees(np.arctan(slope))
    if angle > 5:
        return +1
    elif angle < -5:
        return -1
    return 0
