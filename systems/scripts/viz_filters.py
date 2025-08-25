from __future__ import annotations

"""Centralized logic for visualization filters."""

from dataclasses import dataclass
import math
from typing import Any

from systems.utils import log
from systems.utils.config_loader import get_viz_filters


@dataclass
class VizFilters:
    volatility_min_size: float = 0.0
    pressure_min_size: float = 0.0
    angle_skip_n: int = 1

    _logged: bool = False

    @classmethod
    def from_settings(cls) -> "VizFilters":
        cfg = get_viz_filters()
        filt = cls(**cfg)
        if not cls._logged:
            msg = (
                f"[VIZ_FILTERS] vol_min={filt.volatility_min_size} "
                f"pressure_min={filt.pressure_min_size} "
                f"angle_skip_n={filt.angle_skip_n}"
            )
            log.logic(msg)
            cls._logged = True
        return filt

    @staticmethod
    def _norm(size: Any) -> float:
        try:
            val = float(size)
        except (TypeError, ValueError):
            return 0.0
        if math.isnan(val) or val < 0:
            return 0.0
        return val

    def allow_volatility(self, size: Any) -> bool:
        return self._norm(size) >= self.volatility_min_size

    def allow_pressure(self, size: Any) -> bool:
        return self._norm(size) >= self.pressure_min_size

    def allow_angle(self, idx: int) -> bool:
        return idx % self.angle_skip_n == 0

