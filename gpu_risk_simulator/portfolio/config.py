from dataclasses import dataclass


@dataclass(frozen=True)
class PortfolioConfig:
    tickers: tuple[str, ...]
    period: str = "1y"
    T: float = 1.0
    n_steps: int = 252