from dataclasses import dataclass

import pandas as pd
import numpy as np

from market_data.ticker import annualized_gbm_stats

@dataclass(frozen=True)
class MultiAssetGBMParams:
    tickers: tuple[str, ...]
    x0: tuple[float, ...]
    mu: tuple[float, ...]
    sigma: tuple[float, ...]
    correlation: np.ndarray
    T: float
    n_steps: int


def build_multi_asset_params(
        tickers: tuple[str, ...],
        aligned_history: pd.DataFrame,
        correlation: np.ndarray,
        T: float,
        n_steps: int
) -> MultiAssetGBMParams:
    """Build GBM parameters for multiple assets from aligned price history.

    aligned_history: DataFrame where each column is a ticker's price series.
    """
    x0s, mus, sigmas = [], [], []
    for ticker in tickers:
        x0, mu, sigma = annualized_gbm_stats(aligned_history[ticker])
        x0s.append(x0)
        mus.append(mu)
        sigmas.append(sigma)

    return MultiAssetGBMParams(
        tickers=tickers,
        x0=tuple(x0s),
        mu=tuple(mus),
        sigma=tuple(sigmas),
        correlation=correlation,
        T=T,
        n_steps=n_steps
    )