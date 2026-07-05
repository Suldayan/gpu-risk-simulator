from unittest.mock import patch

from market_data.ticker import TickerParams
from simulation.pipeline import simulate_from_ticker


@patch("simulation.pipeline.create_engine")
@patch("simulation.pipeline.estimate_ticker_params")
def test_simulate_from_ticker_passes_engine_choice(mock_estimate, mock_create_engine):
    from brownian.execution import HostEngine
    mock_estimate.return_value = TickerParams(ticker="AAPL", x0=100.0, mu=0.05, sigma=0.2)
    mock_create_engine.return_value = HostEngine(seed=42)

    simulate_from_ticker("AAPL", T=1.0, n_steps=252, n_paths=500, engine="host")

    mock_create_engine.assert_called_once_with(kind="host", n_paths=500)