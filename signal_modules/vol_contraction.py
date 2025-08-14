NAME = "vol_contraction"
LOOKBACK = 20

def calculate(df, i):
    if i < LOOKBACK:
        return 0
    ranges = df["high"] - df["low"]
    recent_vol = ranges.iloc[i-5:i].mean()
    past_vol = ranges.iloc[i-LOOKBACK:i-5].mean()
    if past_vol == 0:
        return 0
    if recent_vol < past_vol * 0.5:
        if df["close"].iloc[i] > df["open"].iloc[i]:
            return +1
        elif df["close"].iloc[i] < df["open"].iloc[i]:
            return -1
    return 0
