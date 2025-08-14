NAME = "vol_spike"
LOOKBACK = 20

def calculate(df, i):
    if i < LOOKBACK:
        return 0
    avg_vol = df["volume"].iloc[i-LOOKBACK:i].mean()
    if avg_vol == 0:
        return 0
    if df["volume"].iloc[i] > avg_vol * 2:
        if df["close"].iloc[i] < df["open"].iloc[i]:
            return +1
        else:
            return -1
    return 0
