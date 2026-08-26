import time

import numpy as np

import logging
from gpu_risk_simulator.brownian.errors import GBMNumericalError
from gpu_risk_simulator.brownian.execution import ExecutionEngine, HostEngine, create_engine
from gpu_risk_simulator.brownian.models import GBMParams

logger = logging.getLogger(__name__)


class GeometricBrownianMotion:
    """Simulates correlated GBM price paths for any number of assets."""

    def __init__(
        self,
        params: GBMParams,
        engine: ExecutionEngine | str | None = None,
        seed: int | None = None,
    ) -> None:
        self.params = params
        self._n_assets = len(params.tickers)

        if isinstance(engine, str):
            self.engine: ExecutionEngine = create_engine(kind=engine, seed=seed)
        else:
            self.engine = engine or HostEngine(seed=seed)

        self.dt = params.T / params.n_steps
        self.times = self.engine.linspace(0, params.T, params.n_steps + 1)

        self._drift_coeffs = np.array(params.mu) - 0.5 * np.array(params.sigma) ** 2
        self._sigmas = np.array(params.sigma)
        self._x0s = np.array(params.x0)

        self._L = np.linalg.cholesky(params.correlation) if self._n_assets > 1 else None

        self._buffer_n_paths = None
        self._shocks = None
        self._bm = None
        self._paths = None

    def _ensure_buffers(self, n_paths: int):
        if self._buffer_n_paths == n_paths:
            return

        n_steps = self.params.n_steps
        a = self._n_assets

        self._shocks = self.engine.empty((n_paths, n_steps, a))
        self._bm = self.engine.empty((n_paths, n_steps + 1, a))
        self._paths = self.engine.empty((n_paths, n_steps + 1, a))
        self._buffer_n_paths = n_paths

    def _apply_correlation(self):
        if self._L is None:
            return

        shocks_host = self.engine.to_host(self._shocks)
        correlated = shocks_host @ self._L.T

        if isinstance(self.engine, HostEngine):
            self._shocks = correlated
        else:
            import cupy as cp
            self._shocks = cp.asarray(correlated)

    def _check_finite(self, arr):
        if not self.engine.all_finite(arr):
            p = self.params
            raise GBMNumericalError(
                f"Non-finite values detected. "
                f"mu={p.mu}, sigma={p.sigma}, T={p.T}"
            )

    def simulate_paths(self, n_paths: int, copy_result: bool = True):
        if not isinstance(n_paths, int) or n_paths < 1:
            raise ValueError(f"n_paths must be positive, got {n_paths!r}")

        start = time.perf_counter()
        self._ensure_buffers(n_paths)
        p = self.params

        self.engine.standard_normal(
            (n_paths, p.n_steps, self._n_assets), out=self._shocks
        )
        self._shocks *= np.sqrt(self.dt)

        self._apply_correlation()

        self._bm[:, 0, :] = 0.0
        self.engine.cumsum(self._shocks, axis=1, out=self._bm[:, 1:, :])

        for i in range(self._n_assets):
            drift = self._drift_coeffs[i] * self.times
            diffusion = self._sigmas[i] * self._bm[:, :, i]
            temp = drift + diffusion
            self.engine.exp(temp, out=temp)
            self._paths[:, :, i] = temp * self._x0s[i]

        self._check_finite(self._paths)

        elapsed = time.perf_counter() - start
        logger.info(
            "%d assets x %d paths x %d steps in %.4fs",
            self._n_assets, n_paths, p.n_steps, elapsed,
        )

        result = self._paths.copy() if copy_result else self._paths
        result = self.engine.to_host(result)
        return np.transpose(result, (2, 0, 1))