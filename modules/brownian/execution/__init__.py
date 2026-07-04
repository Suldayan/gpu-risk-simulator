"""Public interface for GBM execution engines."""

from brownian.execution.base import ExecutionEngine
from brownian.execution.host import HostEngine

__all__ = ["ExecutionEngine", "HostEngine", "create_engine"]


def create_engine(kind: str = "host", seed: int | None = None) -> ExecutionEngine:
    """Construct an execution engine by name.

    Parameters
    ----------
    kind : str
        "host" for CPU/NumPy, "device" for GPU/CuPy.
    seed : int | None
        Optional seed for reproducible random number generation.
    """
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