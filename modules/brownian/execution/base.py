from typing import TypeVar, Protocol
import numpy as np
import numpy.typing as npt

# A purely abstract type representing the engine's underlying container type
ArrayT = TypeVar("ArrayT")


class ExecutionEngine(Protocol[ArrayT]):
    """Unified interface for executing array operations across hardware targets."""

    def empty(self, shape: tuple[int, ...]) -> ArrayT: ...

    def linspace(self, start: float, stop: float, num: int) -> ArrayT: ...

    def standard_normal(self, shape: tuple[int, ...], out: ArrayT) -> None: ...

    def cumsum(self, arr: ArrayT, axis: int, out: ArrayT) -> None: ...

    def exp(self, arr: ArrayT, out: ArrayT) -> None: ...

    def all_finite(self, arr: ArrayT) -> bool: ...

    def to_host(self, arr: ArrayT) -> npt.NDArray[np.float64]:
        """Exfiltrate data to standard system memory as a standard NumPy array."""
        ...
