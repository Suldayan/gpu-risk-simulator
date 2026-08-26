import pytest

from gpu_risk_simulator.market_data.ticker import price_history_cache


@pytest.fixture(autouse=True)
def clear_price_cache():
    price_history_cache.clear()
    yield