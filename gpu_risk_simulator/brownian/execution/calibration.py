"""Runtime calibration of the host/device crossover threshold."""

import time
import logging

from brownian.execution import HostEngine


logger = logging.getLogger(__name__)


def gpu_available() -> bool:
    try:
        import cupy as cp
        return cp.cuda.runtime.getDeviceCount() > 0
    except Exception:
        return False

def _time_engine(engine, n_paths: int, n_steps: int, n_runs: int = 3) -> float:
    """Time n_runs rounds of the array ops GBM simulation relies on; return the average."""
    increments = engine.empty((n_paths, n_steps))
    bm_paths = engine.empty((n_paths, n_steps + 1))
    paths_buffer = engine.empty((n_paths, n_steps + 1))

    def run_once() -> None:
        engine.standard_normal((n_paths, n_steps), out=increments)
        bm_paths[:, 0] = 0.0
        engine.cumsum(increments, axis=1, out=bm_paths[:, 1:])
        engine.exp(bm_paths, out=paths_buffer)

    run_once()  # warm-up

    times = []
    for _ in range(n_runs):
        start = time.perf_counter()
        run_once()
        times.append(time.perf_counter() - start)

    return sum(times) / len(times)

def calibrate_threshold(
    n_steps: int = 252,
    candidates: tuple[int, ...] = (10, 100, 1_000, 10_000),
) -> int:
    """Measure this machine's host/device crossover point for GBM simulation.

    Uses fixed, arbitrary GBMParams purely to drive a representative
    workload — the specific x0/mu/sigma values don't affect timing, only
    n_paths and n_steps do. This does not use or affect any user-supplied
    simulation parameters; it's a standalone hardware probe.

    Raises
    ------
    RuntimeError
        If a device (GPU/cupy) engine is unavailable on this machine.
    """
    if not gpu_available():
        logger.info("No CUDA-capable GPU detected; defaulting to host-only.")
        return int("inf")  # nothing will ever meet/exceed this -> always host

    try:
        from brownian.execution.device import DeviceEngine
    except ImportError as e:
        raise RuntimeError(
            "Calibration requires a GPU/cupy-capable device engine. "
            "Install cupy, or set GBM_AUTO_THRESHOLD to skip calibration."
        ) from e

    host = HostEngine(seed=42)
    device = DeviceEngine(seed=42)

    for n_paths in candidates:
        host_time = _time_engine(host, n_paths, n_steps)
        device_time = _time_engine(device, n_paths, n_steps)

        logger.info(
            "Calibration n_paths=%d: host=%.4fs device=%.4fs", n_paths, host_time, device_time
        )

        if device_time < host_time:
            return n_paths

    return candidates[-1]