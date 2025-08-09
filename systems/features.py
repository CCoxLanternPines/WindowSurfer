import numpy as np


def _drawdown(prices: np.ndarray) -> float:
    peak = np.maximum.accumulate(prices)
    drawdowns = (peak - prices) / peak
    return float(drawdowns.max()) if drawdowns.size else 0.0


def _slope(prices: np.ndarray) -> float:
    x = np.arange(len(prices))
    y = np.log(prices)
    x_mean = x.mean()
    y_mean = y.mean()
    cov = ((x - x_mean) * (y - y_mean)).sum()
    var = ((x - x_mean) ** 2).sum()
    return float(cov / var) if var else 0.0


def compute_window_features(prices: np.ndarray, win_len: int) -> np.ndarray:
    """Compute rolling window statistical features on ``prices``.

    Parameters
    ----------
    prices:
        Array of close prices ordered chronologically.
    win_len:
        Size of the rolling window.
    """
    if win_len <= 1 or win_len > len(prices):
        return np.empty((0, 8))
    feats = []
    for i in range(win_len, len(prices) + 1):
        window = prices[i - win_len : i]
        returns = np.diff(window) / window[:-1]
        ret_mean = returns.mean()
        ret_std = returns.std(ddof=1) if returns.size > 1 else 0.0
        ac1 = (
            np.corrcoef(returns[1:], returns[:-1])[0, 1]
            if returns.size > 2 and ret_std > 0
            else 0.0
        )
        center = returns - ret_mean
        m3 = np.mean(center ** 3) if returns.size > 0 else 0.0
        m4 = np.mean(center ** 4) if returns.size > 0 else 0.0
        skew = m3 / (ret_std ** 3) if ret_std else 0.0
        kurt = m4 / (ret_std ** 4) if ret_std else 0.0
        dd = _drawdown(window)
        sl = _slope(window)
        # vol of vol: std of rolling std with subwindow 5
        if returns.size >= 5:
            sub = [returns[j : j + 5].std(ddof=1) for j in range(len(returns) - 4)]
            vol_of_vol = np.std(sub, ddof=1) if len(sub) > 1 else 0.0
        else:
            vol_of_vol = 0.0
        feats.append(
            [ret_mean, ret_std, ac1, skew, kurt, dd, sl, vol_of_vol]
        )
    return np.array(feats, dtype=float)


def zscore_features(X: np.ndarray, stats: dict | None = None) -> tuple[np.ndarray, dict]:
    """Apply z-score normalization to ``X``.

    If ``stats`` is ``None`` the mean and std are computed from ``X`` and
    returned alongside the transformed array.
    """
    if stats is None:
        mean = X.mean(axis=0)
        std = X.std(axis=0)
        stats = {"mean": mean.tolist(), "std": std.tolist()}
    else:
        mean = np.array(stats["mean"])
        std = np.array(stats["std"])
    std[std == 0] = 1
    Xn = (X - mean) / std
    return Xn, stats
