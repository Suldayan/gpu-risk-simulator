"""Domain models for a user's stock holdings and portfolio."""

from dataclasses import dataclass

from portfolio.errors import InvalidHoldingError, InvalidPortfolioError


# portfolio/models.py — extending what you already have

@dataclass(frozen=True)
class Holding:
    """A user-declared position: some shares of one ticker.

    cost_basis is optional — the total dollars originally invested, if known.
    Used only for reporting unrealized P&L; simulation itself only needs
    ticker + shares.
    """
    ticker: str
    shares: float
    cost_basis: float | None = None

    def __post_init__(self) -> None:
        if not self.ticker or not self.ticker.strip():
            raise InvalidHoldingError(f"ticker must be non-empty, got {self.ticker!r}")
        if self.shares <= 0:
            raise InvalidHoldingError(f"shares must be positive, got {self.shares}")
        if self.cost_basis is not None and self.cost_basis <= 0:
            raise InvalidHoldingError(f"cost_basis must be positive if given, got {self.cost_basis}")

    @classmethod
    def from_dollar_amount(cls, ticker: str, dollar_amount: float, price: float) -> "Holding":
        if dollar_amount <= 0:
            raise InvalidHoldingError(f"dollar_amount must be positive, got {dollar_amount}")
        if price <= 0:
            raise InvalidHoldingError(f"price must be positive, got {price}")
        shares = dollar_amount / price
        return cls(ticker=ticker, shares=shares, cost_basis=dollar_amount)


@dataclass(frozen=True)
class HoldingSnapshot:
    """A Holding enriched with a current market price (this is your
    'StockSummary' — ticker + shares + investment amount + live value)."""
    holding: Holding
    current_price: float

    @property
    def current_value(self) -> float:
        return self.holding.shares * self.current_price

    @property
    def unrealized_pnl(self) -> float | None:
        if self.holding.cost_basis is None:
            return None
        return self.current_value - self.holding.cost_basis


@dataclass(frozen=True)
class Portfolio:
    """A collection of holdings, one per distinct ticker.

    Attributes
    ----------
    holdings : tuple[Holding, ...]
        The individual positions making up this portfolio. Must be
        non-empty, with no duplicate tickers.
    """

    holdings: tuple[Holding, ...]

    def __post_init__(self) -> None:
        if not self.holdings:
            raise InvalidPortfolioError("Portfolio must contain at least one holding")

        tickers = [h.ticker for h in self.holdings]
        if len(tickers) != len(set(tickers)):
            duplicates = {t for t in tickers if tickers.count(t) > 1}
            raise InvalidPortfolioError(f"Duplicate ticker(s) in portfolio: {duplicates}")

    @property
    def tickers(self) -> tuple[str, ...]:
        """Tickers held in this portfolio, in holding order."""
        return tuple(h.ticker for h in self.holdings)