"""Fetches and aligns historical price data across a portfolio's holdings."""

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd

from market_data.ticker import fetch_price_history
from market_data.errors import TickerNotFoundError, InsufficientDataError
from portfolio.errors import PortfolioError
from portfolio.models import Portfolio

logger = logging.getLogger(__name__)

DEFAULT_MAX_WORKERS = 5
MIN_OVERLAPPING_DATES = 30


def _fetch_all_histories(
    tickers: tuple[str, ...], period: str, max_workers: int
) -> dict[str, pd.Series]:
    """Fetch each ticker's closing-price history concurrently.

    Returns a dict keyed by ticker — insensitive to completion order.
    """
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_ticker = {
            executor.submit(fetch_price_history, ticker, period): ticker
            for ticker in tickers
        }

        histories: dict[str, pd.Series] = {}
        for future in as_completed(future_to_ticker):
            ticker = future_to_ticker[future]
            try:
                hist = future.result()
                histories[ticker] = hist["Close"]
            except (TickerNotFoundError, InsufficientDataError) as e:
                logger.error("History fetch failed for %s: %s", ticker, e)
                raise PortfolioError(f"Failed to fetch history for '{ticker}': {e}") from e

    return histories


def _align_histories(histories: dict[str, pd.Series], tickers: tuple[str, ...]) -> pd.DataFrame:
    """Combine per-ticker series into one DataFrame, in portfolio order,
    keeping only dates where every ticker has data.
    """
    ordered_series = [histories[ticker] for ticker in tickers]
    return pd.concat(ordered_series, axis=1, keys=tickers).dropna(how="any")


def _validate_aligned_history(aligned: pd.DataFrame, tickers: tuple[str, ...]) -> None:
    """Raise PortfolioError if the aligned history is empty or too sparse."""
    if aligned.empty:
        raise PortfolioError(f"No overlapping trading dates found across tickers: {tickers}")
    if len(aligned) < MIN_OVERLAPPING_DATES:
        raise PortfolioError(
            f"Only {len(aligned)} overlapping dates across tickers, need at least {MIN_OVERLAPPING_DATES}"
        )


def fetch_portfolio_history(
    portfolio: Portfolio, period: str = "1y", max_workers: int = DEFAULT_MAX_WORKERS
) -> pd.DataFrame:
    """Fetch aligned closing-price history for every ticker in a portfolio.

    Returns
    -------
    pd.DataFrame
        Index: dates. Columns: tickers. Values: closing prices.

    Raises
    ------
    PortfolioError
        If any ticker's fetch fails, or the aligned result has too few
        overlapping dates to be useful.
    """
    logger.info("Fetching aligned history for portfolio: %s (period=%s)", portfolio.tickers, period)

    histories = _fetch_all_histories(portfolio.tickers, period, max_workers)
    aligned = _align_histories(histories, portfolio.tickers)
    _validate_aligned_history(aligned, portfolio.tickers)

    logger.info(
        "Aligned history: %d overlapping dates across %d tickers", len(aligned), len(portfolio.tickers)
    )
    return aligned