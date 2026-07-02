import numpy as np

from brownian.gbm import GeometricBrownianMotion, GBMParams
from brownian.errors import GBMParameterError, GBMNumericalError
from market_data.ticker import TickerParams, estimate_ticker_params
from market_data.errors import TickerNotFoundError, InsufficientDataError
from simulation.errors import SimulationError


def simulate_from_ticker(
    ticker: str,
    T: float,
    n_steps: int,
    n_paths: int,
    period: str = "1y",
) -> np.ndarray:
    try:
        tp: TickerParams = estimate_ticker_params(ticker=ticker, period=period)
    except (TickerNotFoundError, InsufficientDataError) as e:
        raise SimulationError(f"Failed to estimate parameters for '{ticker}': {e}") from e

    try:
        gbm_params = GBMParams(x0=tp.x0, mu=tp.mu, sigma=tp.sigma, T=T, n_steps=n_steps)
        gbm = GeometricBrownianMotion(gbm_params)
        return gbm.simulate_paths(n_paths)
    except (GBMParameterError, GBMNumericalError) as e:
        raise SimulationError(f"Simulation failed for '{ticker}': {e}") from e