import pytest
from unittest.mock import patch
import pandas as pd

from market_data.ticker import TickerParams, fetch_price_history, compute_gbm_params
from market_data.errors import (
    TickerParameterError,
    TickerNotFoundError,
    InsufficientDataError,
)


@pytest.fixture
def default_params():
    return TickerParams(ticker="AAPL", x0=1.0, mu=0.1, sigma=0.2)


def test_valid_construction(default_params):
    assert default_params.ticker == "AAPL"
    assert default_params.x0 == 1.0
    assert default_params.mu == 0.1
    assert default_params.sigma == 0.2


@patch("market_data.ticker.yf.Ticker")
def test_fetch_returns_history_on_success(mock_ticker):
    df = pd.DataFrame({"Close": [100.0 + i for i in range(35)]})
    mock_ticker.return_value.history.return_value = df
    result = fetch_price_history("AAPL")
    pd.testing.assert_frame_equal(result, df)


def test_rejects_non_positive_x0():
    with pytest.raises(TickerParameterError, match="x0 must be positive"):
        TickerParams(ticker="AAPL", x0=0.0, mu=0.1, sigma=0.2)


def test_rejects_non_positive_sigma():
    with pytest.raises(TickerParameterError, match="sigma must be positive"):
        TickerParams(ticker="AAPL", x0=1.0, mu=0.1, sigma=0.0)


def test_allows_negative_mu():
    params = TickerParams(ticker="AAPL", x0=1.0, mu=-0.05, sigma=0.2)
    assert params.mu == -0.05


@patch("market_data.ticker.yf.Ticker")
def test_fetch_raises_when_empty(mock_ticker):
    mock_ticker.return_value.history.return_value = pd.DataFrame()
    with pytest.raises(TickerNotFoundError):
        fetch_price_history("FAKETICKER")


@patch("market_data.ticker.yf.Ticker")
def test_fetch_raises_when_insufficient_data(mock_ticker):
    mock_ticker.return_value.history.return_value = pd.DataFrame({"Close": [100.0] * 10})
    with pytest.raises(InsufficientDataError):
        fetch_price_history("AAPL")


@patch("market_data.ticker.yf.Ticker")
def test_fetch_succeeds_at_exactly_30_rows(mock_ticker):
    mock_ticker.return_value.history.return_value = pd.DataFrame({"Close": [100.0] * 30})
    result = fetch_price_history("AAPL")
    assert len(result) == 30

@patch("market_data.ticker.yf.Ticker")
def test_fetch_fails_at_29_rows(mock_ticker):
    mock_ticker.return_value.history.return_value = pd.DataFrame({"Close": [100.0] * 29})
    with pytest.raises(InsufficientDataError):
        fetch_price_history("AAPL")


def test_compute_gbm_params_raises_on_zero_volatility():
    df = pd.DataFrame({"Close": [100.0] * 10})
    with pytest.raises(TickerParameterError):
        compute_gbm_params("AAPL", df)


@patch("market_data.ticker.yf.Ticker")
def test_fetch_retries_on_connection_error_then_succeeds(mock_ticker):
    df = pd.DataFrame({"Close": [100.0] * 35})
    mock_ticker.return_value.history.side_effect = [ConnectionError, ConnectionError, df]
    result = fetch_price_history("AAPL")
    pd.testing.assert_frame_equal(result, df)
    assert mock_ticker.return_value.history.call_count == 3


@patch("market_data.ticker.yf.Ticker")
def test_fetch_does_not_retry_on_ticker_not_found(mock_ticker):
    mock_ticker.return_value.history.return_value = pd.DataFrame()
    with pytest.raises(TickerNotFoundError):
        fetch_price_history("FAKETICKER")
    assert mock_ticker.return_value.history.call_count == 1