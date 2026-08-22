import pytest

from market_data.ticker import _price_history_cache


@pytest.fixture(autouse=True)
def clear_price_cache():
    _price_history_cache.clear()
    yield