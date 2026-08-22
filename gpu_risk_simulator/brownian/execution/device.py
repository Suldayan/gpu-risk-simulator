import cupy as cp
import cupy.typing as cpt
import numpy as np
import numpy.typing as npt


class DeviceEngine:
    """Executes mathematical operations on the GPU using VRAM via CuPy."""

    def __init__(self, seed: int | None = None) -> None:
        self._rng = cp.random.default_rng(seed)

    @staticmethod
    def empty(shape: tuple[int, ...]) -> cpt.NDArray[cp.float64]:
        return cp.empty(shape, dtype=cp.float64)

    @staticmethod
    def linspace(start: float, stop: float, num: int) -> cpt.NDArray[cp.float64]:
        return cp.linspace(start, stop, num)

    def standard_normal(self, shape: tuple[int, ...], out: cpt.NDArray[cp.float64]) -> None:
        self._rng.standard_normal(shape, out=out)

    @staticmethod
    def cumsum(arr: cpt.NDArray[cp.float64], axis: int, out: cpt.NDArray[cp.float64]) -> None:
        cp.cumsum(arr, axis=axis, out=out)

    @staticmethod
    def exp(arr: cpt.NDArray[cp.float64], out: cpt.NDArray[cp.float64]) -> None:
        cp.exp(arr, out=out)

    @staticmethod
    def all_finite(arr: cpt.NDArray[cp.float64]) -> bool:
        return bool(cp.all(cp.isfinite(arr)))

    @staticmethod
    def to_host(arr: cpt.NDArray[cp.float64]) -> npt.NDArray[np.float64]:
        return cp.asnumpy(arr)