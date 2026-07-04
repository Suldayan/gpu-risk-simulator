import numpy as np
import numpy.typing as npt


class HostEngine:
    """Executes mathematical operations locally on the CPU using system memory."""

    def __init__(self, seed: int | None = None) -> None:
        self._rng = np.random.default_rng(seed)

    @staticmethod
    def empty(shape: tuple[int, ...]) -> npt.NDArray[np.float64]:
        return np.empty(shape, dtype=np.float64)

    @staticmethod
    def linspace(start: float, stop: float, num: int) -> npt.NDArray[np.float64]:
        return np.linspace(start, stop, num)

    def standard_normal(self, shape: tuple[int, ...], out: npt.NDArray[np.float64]) -> None:
        self._rng.standard_normal(shape, out=out)

    @staticmethod
    def cumsum(arr: npt.NDArray[np.float64], axis: int, out: npt.NDArray[np.float64]) -> None:
        np.cumsum(arr, axis=axis, out=out)

    @staticmethod
    def exp(arr: npt.NDArray[np.float64], out: npt.NDArray[np.float64]) -> None:
        np.exp(arr, out=out)

    @staticmethod
    def all_finite(arr: npt.NDArray[np.float64]) -> bool:
        return bool(np.all(np.isfinite(arr)))

    @staticmethod
    def to_host(arr: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        return arr