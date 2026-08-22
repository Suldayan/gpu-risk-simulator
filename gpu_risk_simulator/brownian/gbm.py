"""Geometric Brownian Motion simulation for asset price paths."""

import logging
import time
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from brownian.errors import GBMNumericalError, GBMParameterError
from brownian.execution import ExecutionEngine, HostEngine, create_engine

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class GBMParams:
    """Parameters for a Geometric Brownian Motion simulation.

    Attributes
    ----------
    x0 : float
        Initial asset price. Must be positive.
    mu : float
        Annualized drift (expected return).
    sigma : float
        Annualized volatility. Must be positive.
    T : float
        Time horizon, in years. Must be positive.
    n_steps : int
        Number of discrete time steps. Must be at least 1.
    """

    x0: float
    mu: float
    sigma: float
    T: float
    n_steps: int

    def __post_init__(self) -> None:
        if self.x0 <= 0:
            raise GBMParameterError(f"x0 must be positive, got {self.x0}")
        if self.sigma <= 0:
            raise GBMParameterError(f"sigma must be positive, got {self.sigma}")
        if self.T <= 0:
            raise GBMParameterError(f"T must be positive, got {self.T}")
        if self.n_steps < 1:
            raise GBMParameterError(f"n_steps must be >= 1, got {self.n_steps}")


class GeometricBrownianMotion:
    """Simulates asset price paths under Geometric Brownian Motion.

    The execution engine (CPU/NumPy by default, GPU/CuPy optionally) is
    injected at construction time — either as an `ExecutionEngine` instance,
    or as a string ("host" / "device") resolved via `create_engine`.

    Internal scratch arrays are lazily allocated and reused across calls to
    `simulate_paths` when `n_paths` is unchanged. `simulate_paths(...,
    copy_result=False)` returns a view into buffers that are overwritten on
    the next call — only use it when the result is fully consumed before
    the next call.
    """

    def __init__(
        self,
        params: GBMParams,
        engine: ExecutionEngine | str | None = None,
        seed: int | None = None,
    ) -> None:
        self.params = params

        if isinstance(engine, str):
            self.engine: ExecutionEngine = create_engine(kind=engine, seed=seed)
        else:
            self.engine = engine or HostEngine(seed=seed)

        self.dt: float = params.T / params.n_steps
        self.times: npt.NDArray[np.float64] = self.engine.linspace(0, params.T, params.n_steps + 1)

        self._buffer_n_paths: int | None = None
        self._increments: npt.NDArray[np.float64] | None = None
        self._bm_paths: npt.NDArray[np.float64] | None = None
        self._paths_buffer: npt.NDArray[np.float64] | None = None

        logger.debug(
            "GBM initialized: T=%.4f n_steps=%d dt=%.6f engine=%s",
            params.T, params.n_steps, self.dt, type(self.engine).__name__,
        )

    def _ensure_buffers(self, n_paths: int) -> None:
        if self._buffer_n_paths == n_paths:
            return

        n_steps = self.params.n_steps
        self._increments = self.engine.empty((n_paths, n_steps))
        self._bm_paths = self.engine.empty((n_paths, n_steps + 1))
        self._paths_buffer = self.engine.empty((n_paths, n_steps + 1))
        self._buffer_n_paths = n_paths

        logger.debug("Allocated buffers for n_paths=%d, n_steps=%d", n_paths, n_steps)

    def _check_finite(self, paths: npt.NDArray[np.float64]) -> None:
        if not self.engine.all_finite(paths):
            p = self.params
            raise GBMNumericalError(
                f"Simulated paths contain non-finite values. "
                f"Consider reducing mu={p.mu}, sigma={p.sigma}, or T={p.T}."
            )

    def simulate_paths(self, n_paths: int, copy_result: bool = True) -> npt.NDArray[np.float64]:
        """Simulate multiple GBM price paths in a single vectorized call.

        Always returns a plain NumPy array, regardless of engine — GPU
        results are copied off-device automatically via `engine.to_host`.
        """
        if not isinstance(n_paths, int) or n_paths < 1:
            raise ValueError(f"n_paths must be a positive int, got {n_paths!r}")

        start = time.perf_counter()
        self._ensure_buffers(n_paths)
        p = self.params

        self.engine.standard_normal((n_paths, p.n_steps), out=self._increments)
        self._increments *= np.sqrt(self.dt)

        self._bm_paths[:, 0] = 0.0
        self.engine.cumsum(self._increments, axis=1, out=self._bm_paths[:, 1:])

        drift_term = (p.mu - 0.5 * p.sigma**2) * self.times + p.sigma * self._bm_paths
        self.engine.exp(drift_term, out=self._paths_buffer)
        self._paths_buffer *= p.x0

        self._check_finite(self._paths_buffer)

        elapsed = time.perf_counter() - start
        logger.info(
            "Simulated %d paths x %d steps in %.4fs (engine=%s, copy_result=%s)",
            n_paths, p.n_steps, elapsed, type(self.engine).__name__, copy_result,
        )

        result = self._paths_buffer.copy() if copy_result else self._paths_buffer
        return self.engine.to_host(result)