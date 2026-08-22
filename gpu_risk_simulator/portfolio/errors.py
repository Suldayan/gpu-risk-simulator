"""Errors for the portfolio module."""


class PortfolioError(Exception):
    """Base exception for portfolio-related errors."""


class InvalidHoldingError(PortfolioError):
    """Raised when a Holding's data is invalid."""


class InvalidPortfolioError(PortfolioError):
    """Raised when a Portfolio's composition is invalid."""