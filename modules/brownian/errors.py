class GBMError(Exception):
    """Base for GBM-specific errors."""

class GBMParameterError(GBMError):
    """Invalid GBMParams values."""

class GBMNumericalError(GBMError):
    """Simulation produced non-finite values."""
