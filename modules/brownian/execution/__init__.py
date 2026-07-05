"""Public interface for GBM execution engines."""

import os

from brownian.execution.base import ExecutionEngine
from brownian.execution.host import HostEngine
from brownian.execution.config.threshold_config import get_auto_threshold

__all__ = ["ExecutionEngine", "HostEngine", "create_engine"]

AUTO_THRESHOLD = int(os.environ.get("AUTO_THRESHOLD", 1000))


def create_engine(kind: str = "host", seed: int | None = None, n_paths: int | None = None) -> ExecutionEngine:
    if kind == "auto":
        if n_paths is None:
            raise ValueError("kind='auto' requires n_paths to choose an engine")

        threshold = get_auto_threshold()  # cached — cheap after first call
        if n_paths >= threshold:
            try:
                from brownian.execution.device import DeviceEngine
            except ImportError as e:
                raise RuntimeError(
                    "GPU execution requested but cupy is not installed or no CUDA "
                    "device is available. Install cupy or use kind='host' instead."
                ) from e
            return DeviceEngine(seed=seed)
        else:
            return HostEngine(seed=seed)

    if kind == "host":
        return HostEngine(seed=seed)
    if kind == "device":
        try:
            from brownian.execution.device import DeviceEngine
        except ImportError as e:
            raise RuntimeError(
                "GPU execution requested but cupy is not installed or no CUDA "
                "device is available. Install cupy or use kind='host' instead."
            ) from e
        return DeviceEngine(seed=seed)

    raise ValueError(f"Unknown engine kind: {kind!r}")