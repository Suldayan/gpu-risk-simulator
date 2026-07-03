"""Geometric Brownian Motion simulation for asset price paths."""

import logging
import time
from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt
from numpy import dtype, float64, ndarray

from brownian.errors import GBMNumericalError, GBMParameterError

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

    Internal scratch arrays are lazily allocated and reused across calls to
    `simulate_paths` when `n_paths` is unchanged, to avoid repeated large
    allocations. This makes the instance stateful with respect to its
    internal buffers — construction is cheap and side-effect free, but
    `simulate_paths(..., copy_result=False)` returns a view into buffers that
    are overwritten on the next call. See `simulate_paths` for details.
    """

    def __init__(self, params: GBMParams) -> None:
        self.params = params
        self.dt: float = params.T / params.n_steps
        self.times: npt.NDArray[np.float64] = np.linspace(0, params.T, params.n_steps + 1)

        self._rng = np.random.default_rng()

        self._buffer_n_paths: int | None = None
        self._increments: npt.NDArray[np.float64] | None = None
        self._bm_paths: npt.NDArray[np.float64] | None = None
        self._paths_buffer: npt.NDArray[np.float64] | None = None

        logger.debug(
            "GBM initialized: T=%.4f n_steps=%d dt=%.6f",
            params.T, params.n_steps, self.dt,
        )

    def _ensure_buffers(self, n_paths: int) -> None:
        """(Re)allocate scratch buffers only when n_paths has changed."""
        if self._buffer_n_paths == n_paths:
            return

        n_steps = self.params.n_steps
        self._increments = np.empty((n_paths, n_steps), dtype=np.float64)
        self._bm_paths = np.empty((n_paths, n_steps + 1), dtype=np.float64)
        self._paths_buffer = np.empty((n_paths, n_steps + 1), dtype=np.float64)
        self._buffer_n_paths = n_paths

        logger.debug("Allocated buffers for n_paths=%d, n_steps=%d", n_paths, n_steps)

    def _check_finite(self, paths: npt.NDArray[np.float64]) -> None:
        if not np.all(np.isfinite(paths)):
            p = self.params
            raise GBMNumericalError(
                f"Simulated paths contain non-finite values. "
                f"Consider reducing mu={p.mu}, sigma={p.sigma}, or T={p.T}."
            )

    def simulate_path(self) -> npt.NDArray[np.float64]:
        p = self.params
        increments = self._rng.standard_normal(p.n_steps) * np.sqrt(self.dt)
        bm_path = np.concatenate([[0.0], np.cumsum(increments)])
        path = p.x0 * np.exp((p.mu - 0.5 * p.sigma ** 2) * self.times + p.sigma * bm_path)

        self._check_finite(path)
        return path

    def simulate_paths(self, n_paths: int, copy_result: bool = True) -> ndarray[tuple[Any, ...], dtype[float64]] | None:
        if not isinstance(n_paths, int) or n_paths < 1:
            raise ValueError(f"n_paths must be a positive int, got {n_paths!r}")

        start = time.perf_counter()
        self._ensure_buffers(n_paths)
        p = self.params

        self._rng.standard_normal((n_paths, p.n_steps), out=self._increments)
        self._increments *= np.sqrt(self.dt)

        self._bm_paths[:, 0] = 0.0
        np.cumsum(self._increments, axis=1, out=self._bm_paths[:, 1:])

        np.exp(
            (p.mu - 0.5 * p.sigma ** 2) * self.times + p.sigma * self._bm_paths,
            out=self._paths_buffer,
        )
        self._paths_buffer *= p.x0

        self._check_finite(self._paths_buffer)

        elapsed = time.perf_counter() - start
        logger.info(
            "Simulated %d paths x %d steps in %.4fs (copy_result=%s)",
            n_paths, p.n_steps, elapsed, copy_result,
        )

        return self._paths_buffer if copy_result else self._paths_buffer