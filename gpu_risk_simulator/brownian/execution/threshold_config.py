import os
from functools import lru_cache

from brownian.execution.calibration import calibrate_threshold


@lru_cache(maxsize=1)
def get_auto_threshold() -> int:
    """Resolve the host/device crossover threshold: env override, else calibrate."""
    override = os.environ.get("GBM_AUTO_THRESHOLD")
    if override is not None:
        return int(override)
    return calibrate_threshold()