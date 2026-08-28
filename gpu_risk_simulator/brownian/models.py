from dataclasses import dataclass
import numpy as np

from gpu_risk_simulator.brownian.errors import GBMParameterError


@dataclass(frozen=True)
class GBMParams:
    tickers: tuple[str, ...]
    x0: tuple[float, ...]
    mu: tuple[float, ...]
    sigma: tuple[float, ...]
    correlation: np.ndarray
    T: float
    n_steps: int

    def __post_init__(self) -> None:
        for ticker, x0, sigma in zip(self.tickers, self.x0, self.sigma):
            if x0 <= 0:
                raise GBMParameterError(f"x0 must be positive for {ticker}, got {x0}")
            if sigma <= 0:
                raise GBMParameterError(f"sigma must be positive for {ticker}, got {sigma}")
        if self.T <= 0:
            raise GBMParameterError(f"T must be positive, got {self.T}")
        if self.n_steps < 1:
            raise GBMParameterError(f"n_steps must be >= 1, got {self.n_steps}")


def build_gbm_params(
    tickers: tuple[str, ...],
    aligned_history,  # pd.DataFrame
    correlation: np.ndarray,
    T: float,
    n_steps: int,
) -> GBMParams:
    from gpu_risk_simulator.market_data.ticker import annualized_gbm_stats

    x0s, mus, sigmas = [], [], []
    for ticker in tickers:
        x0, mu, sigma = annualized_gbm_stats(aligned_history[ticker])
        x0s.append(x0)
        mus.append(mu)
        sigmas.append(sigma)

    return GBMParams(
        tickers=tickers,
        x0=tuple(x0s),
        mu=tuple(mus),
        sigma=tuple(sigmas),
        correlation=correlation,
        T=T,
        n_steps=n_steps,
    )