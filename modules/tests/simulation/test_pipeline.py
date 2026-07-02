from unittest.mock import patch

import numpy as np
import pytest

from market_data.errors import TickerNotFoundError, InsufficientDataError
from market_data.ticker import TickerParams
from simulation.errors import SimulationError
from simulation.pipeline import simulate_from_ticker


@patch("simulation.pipeline.estimate_ticker_params")
def test_simulate_from_ticker_happy_path(mock_estimate):
    mock_estimate.return_value = TickerParams(ticker="AAPL", x0=100.0, mu=0.05, sigma=0.2)

    paths = simulate_from_ticker("AAPL", T=1.0, n_steps=252, n_paths=50, period="1y")

    assert paths.shape == (50, 253)
    assert np.all(np.isfinite(paths))
    assert np.all(paths[:, 0] == 100.0)


@patch("simulation.pipeline.estimate_ticker_params")
def test_simulate_from_ticker_passes_params_through(mock_estimate):
    mock_estimate.return_value = TickerParams(ticker="MSFT", x0=250.0, mu=0.1, sigma=0.3)

    simulate_from_ticker("MSFT", T=2.0, n_steps=100, n_paths=10, period="6mo")

    mock_estimate.assert_called_once_with(ticker="MSFT", period="6mo")


@patch("simulation.pipeline.estimate_ticker_params")
def test_simulate_from_ticker_wraps_ticker_not_found(mock_estimate):
    mock_estimate.side_effect = TickerNotFoundError("no data for FAKETICKER")

    with pytest.raises(SimulationError):
        simulate_from_ticker("FAKETICKER", T=1.0, n_steps=252, n_paths=10)


@patch("simulation.pipeline.estimate_ticker_params")
def test_simulate_from_ticker_wraps_insufficient_data(mock_estimate):
    mock_estimate.side_effect = InsufficientDataError("only 5 data points")

    with pytest.raises(SimulationError):
        simulate_from_ticker("AAPL", T=1.0, n_steps=252, n_paths=10)


@patch("simulation.pipeline.estimate_ticker_params")
def test_simulate_from_ticker_wraps_gbm_parameter_error(mock_estimate):
    # sigma=0 will fail GBMParams validation once it reaches brownian
    mock_estimate.return_value = TickerParams(ticker="AAPL", x0=100.0, mu=0.05, sigma=1e-10)
    # force a value that IS valid at TickerParams level but invalid once passed
    # as T=0 to GBMParams, to trigger the GBM-side wrap without faking TickerParams itself
    with pytest.raises(SimulationError):
        simulate_from_ticker("AAPL", T=0.0, n_steps=252, n_paths=10)


@patch("simulation.pipeline.estimate_ticker_params")
def test_simulate_from_ticker_preserves_original_exception_as_cause(mock_estimate):
    mock_estimate.side_effect = TickerNotFoundError("no data for FAKETICKER")

    with pytest.raises(SimulationError) as exc_info:
        simulate_from_ticker("FAKETICKER", T=1.0, n_steps=252, n_paths=10)

    assert isinstance(exc_info.value.__cause__, TickerNotFoundError)