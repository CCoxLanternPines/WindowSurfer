NAME = "wick_bias"
LOOKBACK = 1

def calculate(df, i):
    if i < 1:
        return 0
    high, low, close = df.loc[i, ["high", "low", "close"]]
    if high == low:
        return 0
    upper_wick = high - close
    lower_wick = close - low
    if lower_wick / (high - low) > 0.66:
        return +1
    elif upper_wick / (high - low) > 0.66:
        return -1
    return 0
