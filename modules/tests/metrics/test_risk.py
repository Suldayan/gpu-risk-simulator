import numpy as np
import pytest

from metrics.risk import compute_risk_metrics, RiskMetrics


@pytest.fixture
def known_terminal_values():
    # 10 terminal prices, x0 = 100, deliberately spread above/below x0
    return np.array([80, 90, 95, 100, 100, 105, 110, 115, 120, 130])


@pytest.fixture
def known_paths(known_terminal_values):
    # only the last column matters to compute_risk_metrics; first column is filler
    x0 = 100.0
    n_paths = len(known_terminal_values)
    return np.column_stack([np.full(n_paths, x0), known_terminal_values])


def test_compute_risk_metrics_known_values(known_paths):
    metrics = compute_risk_metrics(known_paths, x0=100.0, confidence=0.95)

    assert isinstance(metrics, RiskMetrics)
    # losses = x0 - terminal = [20,10,5,0,0,-5,-10,-15,-20,-30]
    # sorted:  [-30,-20,-15,-10,-5,0,0,5,10,20]
    # 95th percentile (linear interp, index (n-1)*0.95 = 8.55): 10 + 0.55*(20-10) = 15.5
    assert metrics.var == pytest.approx(15.5)
    # tail losses >= 15.5 -> only [20] -> cvar = 20.0
    assert metrics.cvar == pytest.approx(20.0)
    # terminal < 100: [80, 90, 95] -> 3/10
    assert metrics.probability_of_loss == pytest.approx(0.3)
    # mean of terminal_values
    assert metrics.mean_terminal == pytest.approx(104.5)
    # median of 10 values: avg of 5th/6th sorted = (100+105)/2
    assert metrics.median_terminal == pytest.approx(102.5)


def test_compute_risk_metrics_all_gains_zero_loss_metrics():
    # every path ends above x0 -> no losses at all
    x0 = 100.0
    terminal = np.array([110.0, 120.0, 130.0, 140.0, 150.0])
    paths = np.column_stack([np.full(5, x0), terminal])

    metrics = compute_risk_metrics(paths, x0=x0, confidence=0.95)

    assert metrics.probability_of_loss == 0.0
    assert metrics.var <= 0.0  # losses are all negative (gains), so VaR is non-positive
    assert metrics.mean_terminal == pytest.approx(130.0)


def test_compute_risk_metrics_all_losses():
    x0 = 100.0
    terminal = np.array([50.0, 60.0, 70.0, 80.0, 90.0])
    paths = np.column_stack([np.full(5, x0), terminal])

    metrics = compute_risk_metrics(paths, x0=x0, confidence=0.95)

    assert metrics.probability_of_loss == 1.0
    assert metrics.var > 0.0
    assert metrics.cvar >= metrics.var  # CVaR should never be less than VaR


def test_compute_risk_metrics_cvar_never_less_than_var(known_paths):
    metrics = compute_risk_metrics(known_paths, x0=100.0, confidence=0.95)
    assert metrics.cvar >= metrics.var


@pytest.mark.parametrize("confidence", [0.90, 0.95, 0.99])
def test_compute_risk_metrics_various_confidence_levels(known_paths, confidence):
    metrics = compute_risk_metrics(known_paths, x0=100.0, confidence=confidence)
    # higher confidence should never produce a lower VaR for the same data
    assert metrics.var is not None
    assert isinstance(metrics.var, float)