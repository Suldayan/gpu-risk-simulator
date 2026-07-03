import numpy as np
import pytest

from simulation.pipeline import simulate_from_ticker


@pytest.mark.integration
def test_simulate_from_ticker_api():
    """Hits the real Yahoo Finance API through the full pipeline."""
    result = simulate_from_ticker("AAPL", T=1.0, n_steps=252, n_paths=100, period="1y")
    assert result.paths.shape == (100, 253)
    assert np.all(np.isfinite(result.paths))