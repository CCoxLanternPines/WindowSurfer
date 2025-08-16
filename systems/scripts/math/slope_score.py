import math

FLAT_BAND_DEG = 10.0  # degrees, adjustable

def slope_to_angle(slope: float) -> float:
    """Return the angle (in degrees) of ``slope``."""
    return math.degrees(math.atan(slope))


def classify_slope(slope: float, flat_band_deg: float = FLAT_BAND_DEG) -> int:
    """Return -1 for down, 0 for flat, +1 for up."""
    angle = slope_to_angle(slope)
    if -flat_band_deg <= angle <= flat_band_deg:
        return 0
    return 1 if angle > flat_band_deg else -1
