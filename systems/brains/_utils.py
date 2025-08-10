import math


def sma(arr, w):
    res = [math.nan] * len(arr)
    if w <= 0 or len(arr) < w:
        return res
    csum = [0.0]
    for x in arr:
        csum.append(csum[-1] + x)
    for i in range(w, len(arr) + 1):
        res[i - 1] = (csum[i] - csum[i - w]) / w
    return res


def rstd(arr, w):
    res = [math.nan] * len(arr)
    if w <= 0 or len(arr) < w:
        return res
    csum = [0.0]
    csum2 = [0.0]
    for x in arr:
        csum.append(csum[-1] + x)
        csum2.append(csum2[-1] + x * x)
    for i in range(w, len(arr) + 1):
        s = csum[i] - csum[i - w]
        s2 = csum2[i] - csum2[i - w]
        mean = s / w
        var = s2 / w - mean * mean
        res[i - 1] = math.sqrt(var) if var > 0 else 0.0
    return res


def atr(high, low, close, w):
    tr = [high[0] - low[0]]
    for i in range(1, len(close)):
        tr.append(max(high[i], close[i - 1]) - min(low[i], close[i - 1]))
    return sma(tr, w)


def zscore(close, w):
    m = sma(close, w)
    sd = rstd(close, w)
    res = [math.nan] * len(close)
    for i in range(len(close)):
        if sd[i] == 0 or math.isnan(sd[i]) or math.isnan(m[i]):
            res[i] = math.nan
        else:
            res[i] = (close[i] - m[i]) / sd[i]
    return res


def slope(arr, w):
    res = [math.nan] * len(arr)
    for i in range(w, len(arr)):
        res[i] = (arr[i] - arr[i - w]) / w
    return res


def first_hits(close, start_idx, up_pct, dn_pct, max_h):
    start = close[start_idx]
    up = start * (1 + up_pct)
    dn = start * (1 + dn_pct)
    for i in range(1, max_h + 1):
        idx = start_idx + i
        if idx >= len(close):
            break
        price = close[idx]
        if price >= up:
            return "up", i
        if price <= dn:
            return "down", i
    return None, None
