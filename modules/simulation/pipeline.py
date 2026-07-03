import numpy as np

from brownian.gbm import GeometricBrownianMotion, GBMParams
from brownian.errors import GBMParameterError, GBMNumericalError
from market_data.ticker import TickerParams, estimate_ticker_params
from market_data.errors import TickerNotFoundError, InsufficientDataError
from metrics.risk import RiskMetrics
from simulation.errors import SimulationError
from metrics.risk import compute_risk_metrics
from dataclasses import dataclass


@dataclass(frozen=True)
class SimulationResult:
    paths: np.ndarray
    metrics: RiskMetrics


def simulate_from_ticker(
    ticker: str, T: float, n_steps: int, n_paths: int, period: str = "1y"
) -> SimulationResult:
    try:
        tp = estimate_ticker_params(ticker=ticker, period=period)
    except (TickerNotFoundError, InsufficientDataError) as e:
        raise SimulationError(f"Failed to estimate parameters for '{ticker}': {e}") from e

    try:
        gbm_params = GBMParams(x0=tp.x0, mu=tp.mu, sigma=tp.sigma, T=T, n_steps=n_steps)
        gbm = GeometricBrownianMotion(gbm_params)
        paths = gbm.simulate_paths(n_paths)
        metrics = compute_risk_metrics(paths, x0=tp.x0)
        return SimulationResult(paths=paths, metrics=metrics)
    except (GBMParameterError, GBMNumericalError) as e:
        raise SimulationError(f"Simulation failed for '{ticker}': {e}") from e