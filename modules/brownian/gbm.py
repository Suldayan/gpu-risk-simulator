import numpy as np

from dataclasses import dataclass
from brownian.errors import GBMParameterError, GBMNumericalError  # add GBMNumericalError


@dataclass(frozen=True)
class GBMParams:
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
    def __init__(self, params: GBMParams) -> None:
        self.params = params
        self.dt = params.T / params.n_steps
        self.times = np.linspace(0, params.T, params.n_steps + 1)

    def _brownian_path(self) -> np.ndarray:
        increments = np.random.normal(0, np.sqrt(self.dt), self.params.n_steps)
        return np.concatenate([[0], np.cumsum(increments)])

    def simulate_path(self) -> np.ndarray:
        p = self.params
        bm_path = self._brownian_path()
        path = p.x0 * np.exp((p.mu - 0.5 * p.sigma**2) * self.times + p.sigma * bm_path)
        if not np.all(np.isfinite(path)):
            raise GBMNumericalError(
                f"Path contains non-finite values. Consider reducing mu={p.mu}, "
                f"sigma={p.sigma}, or T={p.T}."
            )
        return path

    def simulate_paths(self, n_paths: int) -> np.ndarray:
        if not isinstance(n_paths, int) or n_paths < 1:
            raise ValueError(f"n_paths must be a positive int, got {n_paths!r}")
        p = self.params
        increments = np.random.normal(0, np.sqrt(self.dt), (n_paths, self.params.n_steps))
        bm_paths = np.concatenate([np.zeros((n_paths, 1)), np.cumsum(increments, axis=1)], axis=1)
        return p.x0 * np.exp((p.mu - 0.5 * p.sigma ** 2) * self.times + p.sigma * bm_paths)