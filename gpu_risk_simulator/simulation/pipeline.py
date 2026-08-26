from dataclasses import dataclass

import numpy as np
import pandas as pd

import logging
from gpu_risk_simulator.brownian.errors import GBMParameterError, GBMNumericalError
from gpu_risk_simulator.brownian.gbm import GeometricBrownianMotion
from gpu_risk_simulator.brownian.models import GBMParams, build_gbm_params
from gpu_risk_simulator.portfolio.fetcher import fetch_aligned_history
from gpu_risk_simulator.simulation.errors import SimulationError

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SimulationResult:
    paths: np.ndarray
    params: GBMParams

def _correlate(aligned: pd.DataFrame) -> np.ndarray:
    """Compute correlation matrix from daily returns."""
    returns = aligned.pct_change().dropna()
    return returns.corr().values


def _build_params(
    tickers: tuple[str, ...],
    aligned: pd.DataFrame,
    correlation: np.ndarray,
    T: float,
    n_steps: int,
) -> GBMParams:
    """Estimate GBM parameters from aligned price history."""
    try:
        return build_gbm_params(
            tickers=tickers,
            aligned_history=aligned,
            correlation=correlation,
            T=T,
            n_steps=n_steps,
        )
    except Exception as e:
        raise SimulationError(f"Parameter estimation failed: {e}") from e


def simulate(
    tickers: list[str],
    T: float,
    n_steps: int,
    n_paths: int,
    period: str = "1y",
    engine: str = "auto",
) -> SimulationResult:
    """Simulate correlated GBM paths for any number of assets."""
    logger.info(
        "Simulating %s (T=%.2f, n_steps=%d, n_paths=%d)",
        tickers, T, n_steps, n_paths,
    )

    aligned = fetch_aligned_history(tuple(tickers), period=period)
    correlation = _correlate(aligned)
    params = _build_params(tuple(tickers), aligned, correlation, T, n_steps)

    try:
        gbm = GeometricBrownianMotion(params, engine=engine)
        paths = gbm.simulate_paths(n_paths)
    except (GBMParameterError, GBMNumericalError) as e:
        raise SimulationError(f"Simulation failed: {e}") from e

    logger.info("Simulation complete for %s", tickers)
    return SimulationResult(paths=paths, params=params)