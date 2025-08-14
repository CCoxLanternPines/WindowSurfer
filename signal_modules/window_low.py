NAME = "window_low"
LOOKBACK = 20

def calculate(df, i):
    if i < LOOKBACK:
        return 0
    low_window = df["low"].iloc[i-LOOKBACK+1:i+1].min()
    if df["low"].iloc[i] <= low_window:
        return +1
    return 0
