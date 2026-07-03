"""Project-wide logging configuration."""

import logging


def setup_logging(level: int = logging.INFO, formatter: logging.Formatter | None = None) -> None:
    """Configure the root logger once, at the application's entry point.

    Domain modules should never call this — they only call
    `logging.getLogger(__name__)` and emit records. Only the entry point
    (CLI script, API startup, test harness) decides how those records
    are formatted and where they go.
    """
    handler = logging.StreamHandler()
    handler.setFormatter(
        formatter
        or logging.Formatter(
            "%(asctime)s | %(name)-30s | %(levelname)-8s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    )

    root = logging.getLogger()
    root.setLevel(level)
    root.addHandler(handler)

    logging.getLogger("yfinance").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("peewee").setLevel(logging.WARNING)