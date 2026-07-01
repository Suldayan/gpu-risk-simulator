import pytest

from market_data.ticker import fetch_price_history


@pytest.mark.integration
def test_fetch_price_history():
    """Hits the real Yahoo Finance API. Run explicitly, not in normal test runs."""
    hist = fetch_price_history("AAPL", period="2mo")
    assert not hist.empty
    assert "Close" in hist.columns
    assert len(hist) > 0