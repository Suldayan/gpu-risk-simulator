class TickerError(Exception):
    """Base class for exceptions in this module."""

class TickerParameterError(TickerError):
    """Raised when parameters return incorrect values."""

class TickerNotFoundError(TickerError):
    """Raised when a ticker cannot be found."""

class InsufficientDataError(TickerError):
    """Raised when an insufficient data is encountered."""