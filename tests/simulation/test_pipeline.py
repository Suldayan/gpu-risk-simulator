from unittest.mock import patch, MagicMock

import numpy as np
import pytest

from gpu_risk_simulator.market_data.errors import TickerNotFoundError
from gpu_risk_simulator.simulation.errors import SimulationError
from gpu_risk_simulator.simulation.pipeline import simulate


@patch("gpu_risk_simulator.simulation.pipeline.fetch_aligned_history")
@patch("gpu_risk_simulator.simulation.pipeline.GeometricBrownianMotion")
def test_simulate_happy_path(mock_gbm_class, mock_fetch):
    # Mock the aligned history DataFrame
    mock_fetch.return_value = MagicMock()
    mock_fetch.return_value.pct_change.return_value.dropna.return_value.corr.return_value.values = np.array([[1.0]])

    # Mock GBM to return deterministic paths
    mock_gbm = MagicMock()
    mock_gbm.simulate_paths.return_value = np.ones((1, 50, 253)) * 100.0
    mock_gbm_class.return_value = mock_gbm

    result = simulate(["AAPL"], T=1.0, n_steps=252, n_paths=50, period="1y")

    # Unified shape: (n_assets, n_paths, n_steps+1)
    assert result.paths.shape == (1, 50, 253)
    mock_fetch.assert_called_once()


@patch("gpu_risk_simulator.simulation.pipeline.fetch_aligned_history")
def test_simulate_wraps_fetch_errors(mock_fetch):
    mock_fetch.side_effect = TickerNotFoundError("no data for FAKETICKER")

    with pytest.raises(SimulationError) as exc_info:
        simulate(["FAKETICKER"], T=1.0, n_steps=252, n_paths=10)

    assert isinstance(exc_info.value.__cause__, TickerNotFoundError)