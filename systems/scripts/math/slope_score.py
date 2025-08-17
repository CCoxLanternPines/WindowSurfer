import math


def classify_slope(slope: float, flat_band_deg: float) -> int:
    """Classify slope into trend directions.

    Parameters
    ----------
    slope:
        Slope value as a rate of change.
    flat_band_deg:
        Threshold in degrees below which the slope is considered flat.

    Returns
    -------
    int
        -1 for downward trend, 1 for upward trend, 0 for flat.
    """
    slope_deg = math.degrees(math.atan(slope))
    if slope_deg > flat_band_deg:
        return 1
    if slope_deg < -flat_band_deg:
        return -1
    return 0
