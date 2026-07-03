from dataclasses import dataclass

from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential
from market_data.errors import TickerParameterError, TickerNotFoundError, InsufficientDataError

import logging
import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)

max_attempts = 3
default_period = "1y"

@dataclass(frozen=True)
class TickerParams:
    ticker: str
    x0: float
    mu: float
    sigma: float

    def __post_init__(self) -> None:
        if self.x0 <= 0:
            raise TickerParameterError(f"x0 must be positive, got {self.x0}")
        if self.sigma <= 0:
            raise TickerParameterError(f"sigma must be positive, got {self.sigma}")


@retry(
    retry=retry_if_exception_type((ConnectionError, TimeoutError)),
    stop=stop_after_attempt(max_attempts),
    wait=wait_exponential(multiplier=1, min=1, max=8),
    reraise=True
)
def fetch_price_history(ticker: str, period: str = default_period) -> pd.DataFrame:
    logger.debug("Fetching history for %s (period=%s)", ticker, period)
    hist = yf.Ticker(ticker).history(period=period)

    if hist.empty:
        logger.warning("No data returned for %s", ticker)
        raise TickerNotFoundError(f"No data found for {ticker}")
    if len(hist) < 30:
        logger.warning("Insufficient data for %s: %d rows", ticker, len(hist))
        raise InsufficientDataError(f"Only {len(hist)} data points for {ticker}, need at least 30")

    logger.info("Fetched %d rows for %s", len(hist), ticker)
    return hist

def compute_gbm_params(ticker: str, hist: pd.DataFrame) -> TickerParams:
    returns = hist["Close"].pct_change().dropna()

    x0 = float(hist["Close"].iloc[-1])
    mu = float(returns.mean() * 252)
    sigma = float(returns.std() * (252 ** 0.5))

    logger.debug("Computed params for %s: x0=%.2f mu=%.4f sigma=%.4f", ticker, x0, mu, sigma)

    return TickerParams(ticker=ticker, x0=x0, mu=mu, sigma=sigma)

def estimate_ticker_params(ticker: str, period: str = default_period) -> TickerParams:
    hist = fetch_price_history(ticker, period)
    return compute_gbm_params(ticker, hist)


