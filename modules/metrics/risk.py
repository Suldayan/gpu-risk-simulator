import numpy as np
from dataclasses import dataclass


@dataclass(frozen=True)
class RiskMetrics:
    var: float
    cvar: float
    probability_of_loss: float
    mean_terminal: float
    median_terminal: float


def compute_risk_metrics(
    paths: np.ndarray,
    x0: float,
    confidence: float = 0.95
) -> RiskMetrics:
    terminal_values = paths[:, -1]
    losses = x0 - terminal_values

    var = float(np.percentile(losses, confidence * 100))
    tail_losses = losses[losses >= var]
    cvar = float(tail_losses.mean()) if len(tail_losses) > 0 else var

    return RiskMetrics(
        var=var,
        cvar=cvar,
        probability_of_loss=float(np.mean(terminal_values < x0)),
        mean_terminal=float(terminal_values.mean()),
        median_terminal=float(np.median(terminal_values)),
    )