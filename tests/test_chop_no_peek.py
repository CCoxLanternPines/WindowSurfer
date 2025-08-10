from systems.scripts.chop_policy import PolicyState, compute_features


def run_series(prices):
    state = PolicyState(window=3, slope_gap=1)
    feats = []
    for p in prices:
        candle = {'open': p, 'high': p, 'low': p, 'close': p, 'volume': 0}
        feats.append(compute_features(state, candle))
    return feats


def test_no_peek_features_immutable():
    prices = [1, 2, 3, 4, 5, 6, 7, 8]
    feats1 = run_series(prices)
    baseline = feats1[4]

    prices2 = prices.copy()
    prices2[6] = 100  # modify future price
    feats2 = run_series(prices2)

    compare = feats2[4]
    assert baseline['ma'] == compare['ma']
    assert baseline['slope'] == compare['slope']
    assert baseline['atr_pct'] == compare['atr_pct']
    assert baseline['dev_pct'] == compare['dev_pct']
