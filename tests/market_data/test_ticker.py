import pytest
from unittest.mock import patch
import pandas as pd

from market_data.ticker import TickerParams, fetch_price_history, compute_gbm_params, estimate_ticker_params
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


@pytest.mark.parametrize(
    "history, expected_exception",
    [
        (pd.DataFrame(), TickerNotFoundError),
        (pd.DataFrame({"Close": [100.0] * 10}), InsufficientDataError),
        (pd.DataFrame({"Close": [100.0] * 29}), InsufficientDataError),
    ],
)
@patch("market_data.ticker.yf.Ticker")
def test_fetch_raises_on_bad_data(mock_ticker, history, expected_exception):
    mock_ticker.return_value.history.return_value = history
    with pytest.raises(expected_exception):
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


@patch("market_data.ticker.yf.Ticker")
def test_fetch_gives_up_after_max_attempts(mock_ticker):
   mock_ticker.return_value.history.side_effect = ConnectionError
   with pytest.raises(ConnectionError):
       fetch_price_history("AAPL")
   assert mock_ticker.return_value.history.call_count == 3


@patch("market_data.ticker.yf.Ticker")
def test_fetch_does_not_retry_on_insufficient_data(mock_ticker):
   mock_ticker.return_value.history.return_value = pd.DataFrame({"Close": [100.0] * 10})
   with pytest.raises(InsufficientDataError):
       fetch_price_history("AAPL")
   assert mock_ticker.return_value.history.call_count == 1


@patch("market_data.ticker.yf.Ticker")
def test_estimate_ticker_params_end_to_end(mock_ticker):
   prices = [100.0 + (i % 3) for i in range(35)]
   mock_ticker.return_value.history.return_value = pd.DataFrame({"Close": prices})
   result = estimate_ticker_params("AAPL")
   assert isinstance(result, TickerParams)
   assert result.ticker == "AAPL"
   assert result.x0 == prices[-1]
   assert result.sigma > 0.0


def test_compute_gbm_params_known_values():
    prices = [100.0, 101.0, 99.0, 102.0, 98.0, 103.0, 97.0, 104.0, 96.0, 105.0]
    df = pd.DataFrame({"Close": prices})

    returns = pd.Series(prices).pct_change().dropna()
    expected_mu = returns.mean() * 252
    expected_sigma = returns.std() * (252 ** 0.5)

    params = compute_gbm_params("AAPL", df)

    assert params.x0 == pytest.approx(105.0)
    assert params.mu == pytest.approx(expected_mu)
    assert params.sigma == pytest.approx(expected_sigma)